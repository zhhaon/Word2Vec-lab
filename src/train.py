"""第 5 步：模型训练。

特点：
    * 支持 GPU（RTX 5090）与 CPU 自动选择；请求 GPU 但不可用时给出开卡提醒；
    * 时间预算保护 max_minutes：到点自动优雅停止并保存，保证不超过 6 小时上限；
    * 支持断点续训、TensorBoard、CSV 指标、词向量双格式导出；
    * 后台线程预取数据，GPU 不空转。

一条命令跑完（tiny 配置本地几分钟可完成）：
    python -m src.train --config configs/tiny.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .early_stop import (EarlyStopper, canonical_metric, extract_metric,
                         is_finite)
from .dataset import PairStream
from .model import Word2VecNeg, build_optimizer, estimate_parameters, lr_at_step
from .preprocess import load_processed, preprocess as run_preprocess
from .utils import (Timer, ensure_dir, format_hms, get_logger, human_int,
                    load_config, resolve_path, save_json, set_seed)
from .vectors import save_vectors
from .vocab import NegativeSampler, Vocab


# --------------------------------------------------------------------------
# 梯度裁剪：手写以兼容稀疏梯度（torch 自带的 clip_grad_norm_ 对稀疏梯度不友好）
# --------------------------------------------------------------------------
def grad_total_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad
        if g.is_sparse:
            total += float(g.coalesce().values().pow(2).sum())
        else:
            total += float(g.detach().pow(2).sum())
    return math.sqrt(total)


def clip_grad_norm_(parameters, max_norm: float) -> float:
    total_norm = grad_total_norm(parameters)
    if max_norm and max_norm > 0:
        scale = max_norm / (total_norm + 1e-6)
        if scale < 1.0:
            for p in parameters:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    g = p.grad.coalesce()
                    g._values().mul_(scale)
                    p.grad = g
                else:
                    p.grad.detach().mul_(scale)
    return total_norm


def neg_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """NEG 目标：对 (1+K) 项求和后按 batch 求平均（与论文目标一致）。"""
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="sum") / logits.size(0)


# --------------------------------------------------------------------------
# 数据预取
# --------------------------------------------------------------------------
class Prefetcher:
    """在后台线程里生成样本并转成 GPU 张量，避免 GPU 等 CPU。"""

    def __init__(self, batches, depth: int = 8, device=None, pin: bool = False) -> None:
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        self.device = device
        self.pin = pin
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._worker, args=(batches,), daemon=True)
        self.thread.start()

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if self.pin and self.device is not None and self.device.type == "cuda":
            t = t.pin_memory()
        return t.to(self.device, non_blocking=True)

    def _worker(self, batches) -> None:
        try:
            for epoch, center, context, negatives in batches:
                if self._stop.is_set():
                    break
                self.q.put((
                    epoch,
                    self._to_tensor(center),
                    self._to_tensor(context),
                    None if negatives is None else self._to_tensor(negatives),
                ))
        except Exception as exc:  # noqa: BLE001
            self.q.put(exc)
        finally:
            self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------
# 数据准备
# --------------------------------------------------------------------------
def prepare_data(cfg: Dict[str, Any], logger, auto: bool = True):
    d = cfg["data"]
    source = d["source"]
    lang = d.get("lang", "en")
    min_count = int(d.get("min_count", 5))
    max_tokens = d.get("max_tokens")

    try:
        return load_processed(source, lang, min_count, max_tokens,
                              d.get("processed_dir", "data/processed"))
    except FileNotFoundError as exc:
        if not auto:
            raise
        logger.warning("%s", exc)
        logger.info("自动执行预处理 ...")
        run_preprocess(source, lang, d.get("raw_dir", "data/raw"),
                       d.get("processed_dir", "data/processed"),
                       min_count, d.get("max_vocab"), max_tokens)
        return load_processed(source, lang, min_count, max_tokens,
                              d.get("processed_dir", "data/processed"))


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------
def train(cfg: Dict[str, Any], resume: Optional[str] = None,
          do_eval: bool = True, do_viz: bool = False) -> Path:
    t_cfg, m_cfg = cfg["train"], cfg["model"]
    run_name = cfg.get("run_name", "run")
    out_dir = ensure_dir(resolve_path(f"{t_cfg.get('out_dir', 'runs')}/{run_name}"))
    logger = get_logger("word2vec", str(out_dir / "train.log"))
    set_seed(int(cfg.get("seed", 42)))

    logger.info("=" * 74)
    logger.info("Word2Vec 训练  |  run=%s", run_name)
    logger.info("=" * 74)

    tokens, sent_lengths, vocab, meta = prepare_data(cfg, logger)
    logger.info("语料: %s  词表 %s  句子 %s  <unk> 占比 %.2f%%",
                meta["source"], human_int(meta["vocab_size"]), human_int(meta["num_sentences"]),
                meta["unk_rate"] * 100)

    # ---------------- 设备 ----------------
    from .utils import resolve_device

    device = resolve_device(t_cfg.get("device", "auto"), logger)

    # ---------------- 模型 ----------------
    arch = m_cfg.get("arch", "skipgram")
    dim = int(m_cfg.get("dim", 300))
    window = int(m_cfg.get("window", 5))
    sparse = bool(m_cfg.get("sparse", True))
    n_neg = int(m_cfg.get("negative", 10))

    model = Word2VecNeg(len(vocab), dim=dim, arch=arch, sparse=sparse).to(device)
    logger.info("结构: %s  维度: %d  窗口: ±%d  负样本: %d  稀疏梯度: %s",
                arch, dim, window, n_neg, sparse)
    logger.info("模型规模: %s", estimate_parameters(len(vocab), dim))

    optimizer_name = t_cfg.get("optimizer", "sparseadam")
    if sparse and optimizer_name.lower() in ("adamw", "adam"):
        raise ValueError(
            "sparse=True 时必须搭配 sparseadam / adagrad / sgd（AdamW 不支持稀疏梯度）。"
            "想用 AdamW 请在配置里设 model.sparse=false"
        )
    base_lr = float(t_cfg.get("lr", 0.0025))
    min_lr = float(t_cfg.get("min_lr", 0.0001))
    optimizer = build_optimizer(model, optimizer_name, base_lr,
                                float(t_cfg.get("weight_decay", 0.0)))

    # ---------------- 样本流 ----------------
    stream = PairStream(tokens, sent_lengths, vocab, window=window, arch=arch,
                        subsample=float(m_cfg.get("subsample", 0.0)),
                        seed=int(cfg.get("seed", 42)))
    stream.prepare(0)
    neg_sampler = NegativeSampler(vocab, seed=int(cfg.get("seed", 42)) + 1)

    batch_size = int(t_cfg.get("batch_size", 8192))
    epochs = int(t_cfg.get("epochs", 5))
    steps_per_epoch = max(1, stream.pairs_per_epoch // batch_size)
    total_steps = steps_per_epoch * epochs

    logger.info("每个 epoch: %s token -> 约 %s 个样本对 -> %s 步 (batch=%d)",
                human_int(stream.tokens_per_epoch), human_int(stream.pairs_per_epoch),
                human_int(steps_per_epoch), batch_size)
    logger.info("计划: %d 个 epoch，合计约 %s 步", epochs, human_int(total_steps))

    # ---------------- 恢复 ----------------
    start_epoch, global_step = 0, 0
    if resume:
        ckpt_path = resolve_path(resume)
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt["step"]
            logger.info("已从 %s 恢复：epoch=%d step=%d", ckpt_path, start_epoch, global_step)

    # ---------------- 日志设施 ----------------
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(out_dir / "tb"))
    except Exception:  # noqa: BLE001
        logger.warning("TensorBoard 不可用，跳过（pip install tensorboard 可开启）")

    metrics_path = out_dir / "metrics.csv"
    new_file = not metrics_path.exists()
    metrics_f = open(metrics_path, "a", newline="", encoding="utf-8")
    metrics_w = csv.writer(metrics_f)
    if new_file:
        metrics_w.writerow(["epoch", "step", "loss", "lr", "grad_norm",
                            "pairs_seen", "elapsed_sec", "pairs_per_sec"])

    # ---------------- 训练循环 ----------------
    max_minutes = float(t_cfg.get("max_minutes", 0) or 0)
    deadline = time.time() + max_minutes * 60 if max_minutes > 0 else None
    log_every = int(t_cfg.get("log_every", 500))
    ckpt_every = int(t_cfg.get("ckpt_every_steps", 20000))
    max_grad = float(t_cfg.get("max_grad_norm", 0) or 0)
    prefetch = int(t_cfg.get("prefetch", 8))
    eval_every = int(cfg.get("eval", {}).get("every_epochs", 0) or 0)

    # ---------------- 早停 ----------------
    es_cfg = cfg.get("early_stopping", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    monitor = str(es_cfg.get("monitor", "analogy_accuracy"))
    export_best = bool(es_cfg.get("export_best", True))
    stopper = EarlyStopper(
        patience=int(es_cfg.get("patience", 3)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        min_epochs=int(es_cfg.get("min_epochs", 1)),
    ) if es_enabled else None

    if eval_every > 0 or es_enabled:
        prefetch_eval_sets(cfg, logger)
    if es_enabled:
        logger.info("早停: 开启  监控 %s（越大越好）  耐心 %d  最小间隔 %g  至少 %d 个 epoch",
                    canonical_metric(monitor), stopper.patience,
                    stopper.min_delta, stopper.min_epochs)
    elif eval_every > 0:
        logger.info("早停: 关闭")

    def sample_negatives(center_np: np.ndarray, context_np: np.ndarray) -> Optional[np.ndarray]:
        """按「排除正目标」的规则采负样本（在预取线程里做，不阻塞 GPU）。

        Skip-gram：正目标是上下文词，另外把中心词也禁掉
                   （中心词与正目标是一对真共现，抽成负样本同样是假负样本）；
        CBOW     ：正目标是中心词，另外把整条上下文都禁掉。
        """
        if n_neg <= 0:
            return None
        if arch == "skipgram":
            positive, extra = context_np, center_np
        else:
            positive, extra = center_np, context_np
        return neg_sampler.sample_excluding(
            (len(center_np), n_neg), targets=positive, extra=extra
        )

    def batch_iter() -> Iterator[Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        for ep in range(start_epoch, epochs):
            for c, x in stream.iter_batches(batch_size=batch_size, epoch=ep):
                yield ep, c, x, sample_negatives(c, x)

    prefetcher = Prefetcher(batch_iter(), depth=prefetch, device=device)
    timer = Timer()
    running_loss, run_steps, seen_pairs = 0.0, 0, 0
    stopped_early = False
    last_epoch = start_epoch - 1

    logger.info("-" * 74)
    logger.info("开始训练（max_minutes=%s，到点会自动停止并保存）",
                max_minutes if max_minutes > 0 else "无限制")
    logger.info("负采样约束: 排除本样本的正目标(%s)%s + 排除 <unk>/<pad>",
                "上下文词" if arch == "skipgram" else "中心词",
                " + 排除中心词" if arch == "skipgram" else " + 排除上下文词")
    logger.info("-" * 74)

    stopped_by_es = False
    epoch = start_epoch - 1        # 防止 batch 生成器一个都没产出时下面用到未定义变量

    try:
        for epoch, center, context, negatives in prefetcher:
            # ---------------- epoch 交界处：完整评估 + 早停判断 ----------------
            if epoch != last_epoch:
                finished = last_epoch          # 刚跑完的那个 epoch
                last_epoch = epoch
                if (eval_every > 0 and finished >= start_epoch
                        and finished % eval_every == 0):
                    res, metric = run_epoch_eval(
                        model, vocab, out_dir, finished, cfg, monitor,
                        writer=writer, global_step=global_step)

                    # 先让 stopper 吃掉这次结果，再打日志 ——
                    # 否则摘要里显示的是「上一轮」的最优值与耐心计数，会自相矛盾
                    should_stop = stopper.step(finished, metric) if stopper else False
                    log_eval_summary(logger, finished, res, metric, monitor, stopper)

                    if stopper is not None:
                        if stopper.best_epoch == finished:
                            _save_best(model, vocab, out_dir, finished, metric,
                                       monitor, cfg)
                            logger.info("            指标提升，已保存最优权重 -> %s",
                                        out_dir / "best.pt")
                        if should_stop:
                            logger.warning(
                                "早停：连续 %d 次评估没有超过 %s %.4f，"
                                "在 epoch %d 处停止训练。",
                                stopper.patience,
                                canonical_metric(monitor),
                                stopper.best, finished + 1)
                            stopped_by_es = True
                            break

            if global_step >= total_steps:
                global_step = total_steps - 1  # 学习率调度收尾
            lr = lr_at_step(global_step, total_steps, base_lr, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            bs = center.size(0)
            optimizer.zero_grad(set_to_none=True)
            logits, labels = model(center, context, negatives)
            loss = neg_loss(logits, labels)
            loss.backward()
            gnorm = clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()

            running_loss += float(loss.detach())
            run_steps += 1
            seen_pairs += bs

            if run_steps % log_every == 0 or global_step == 0:
                avg = running_loss / max(1, run_steps)
                elapsed = timer.elapsed
                pps = seen_pairs / max(elapsed, 1e-6)
                remain_steps = total_steps - global_step
                eta = remain_steps / max(pps / batch_size, 1e-9)
                logger.info(
                    "epoch %d/%d | step %s/%s | loss %.4f | lr %.5f | "
                    "gnorm %.2f | %s pairs/s | 已用 %s | 预计还需 %s",
                    epoch + 1, epochs, human_int(global_step), human_int(total_steps),
                    avg, lr, gnorm, human_int(pps), format_hms(elapsed), format_hms(eta),
                )
                if writer:
                    writer.add_scalar("train/loss", avg, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/grad_norm", gnorm, global_step)
                    writer.add_scalar("perf/pairs_per_sec", pps, global_step)
                metrics_w.writerow([epoch + 1, global_step, f"{avg:.6f}", f"{lr:.6f}",
                                    f"{gnorm:.4f}", seen_pairs, f"{elapsed:.1f}", f"{pps:.1f}"])
                metrics_f.flush()
                running_loss, run_steps = 0.0, 0

            if global_step > 0 and global_step % ckpt_every == 0:
                _save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, global_step, cfg)
                logger.info("  已保存检查点 %s (step %s)", out_dir / "last.pt", human_int(global_step))

            global_step += 1

            # 时间预算保护
            if deadline and time.time() > deadline:
                logger.warning("已达到时间预算 %g 分钟，优雅停止并保存。", max_minutes)
                stopped_early = True
                break

        else:
            pass

        # ---------------- 收尾 ----------------
        if deadline and time.time() > deadline and not stopped_early:
            stopped_early = True
    finally:
        prefetcher.close()
        metrics_f.close()

    _save_checkpoint(out_dir / "last.pt", model, optimizer, min(epochs, epoch + 1),
                     global_step, cfg)

    # 早停的意义就是拿到最好的那个模型，所以默认把最优权重导出成 vectors.*
    used_best = False
    if (stopper is not None and export_best and stopper.best_epoch >= 0
            and (out_dir / "best.pt").exists()):
        model.load_state_dict(
            torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
        used_best = True

    npz_path, txt_path = export_vectors(model, vocab, out_dir, m_cfg.get("export", "input"))
    if used_best:
        logger.info("词向量已导出: %s / %s  ← 取自最优 epoch %d（%s = %s）",
                    npz_path.name, txt_path.name, stopper.best_epoch + 1,
                    canonical_metric(monitor), _fmt(stopper.best))
        logger.info("  last.pt 仍保留最后一个 epoch 的状态；最优副本另存为 best.pt / best_vectors.*")
    else:
        logger.info("词向量已导出: %s / %s", npz_path.name, txt_path.name)

    if neg_sampler.n_sampled:
        logger.info("负采样统计: 共 %s 个，其中与正目标/特殊符号冲突而被重抽 %s 个（%.4f%%）",
                    human_int(neg_sampler.n_sampled), human_int(neg_sampler.n_resampled),
                    100 * neg_sampler.resample_rate)

    if stopper is not None:
        save_json(stopper.to_dict(monitor, stopped=stopped_by_es),
                  out_dir / "early_stopping.json")
        logger.info("早停记录: %s", out_dir / "early_stopping.json")

    reasons = []
    if stopped_by_es:
        reasons.append("早停")
    if stopped_early:
        reasons.append("时间预算")
    logger.info("总用时 %s%s", format_hms(timer.elapsed),
                ("（因" + "、".join(reasons) + "提前停止）") if reasons else "")

    if writer:
        writer.flush()
        writer.close()

    # ---------------- 评估 / 可视化 ----------------
    if do_eval:
        try:
            from .evaluate import evaluate_from_npz

            ev = cfg.get("eval", {}) or {}
            res = evaluate_from_npz(npz_path, ev.get("similarity_file"),
                                    ev.get("analogy_file"), int(ev.get("topk", 10)),
                                    logger=logger,
                                    fetch_eval=bool(ev.get("fetch", True)))
            save_json(res, out_dir / "eval.json")
        except Exception as exc:  # noqa: BLE001
            logger.warning("评估失败（不影响训练结果）: %s", exc)

    if do_viz:
        try:
            from .visualize import visualize_run

            visualize_run(cfg, vectors_path=npz_path, logger=logger)
        except Exception as exc:  # noqa: BLE001
            logger.warning("可视化失败（不影响训练结果）: %s", exc)

    logger.info("全部完成。产物目录: %s", out_dir)
    return out_dir


def _save_checkpoint(path, model, optimizer, epoch, step, cfg) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "config": cfg,
    }, path)


def export_vectors(model, vocab: Vocab, out_dir: Path, how: str = "input"):
    mat = model.export_vectors(how).float().cpu().numpy()
    return save_vectors(Path(out_dir) / "vectors", vocab.itos, mat)


# --------------------------------------------------------------------------
# 评估摘要输出
# --------------------------------------------------------------------------
def _fmt(v: Optional[float], nd: int = 4) -> str:
    return "n/a" if not is_finite(v) else f"{float(v):.{nd}f}"


def log_eval_summary(logger, epoch: int, res: Dict, metric: Optional[float],
                     monitor: str, stopper: Optional[EarlyStopper] = None) -> None:
    """中途评估的紧凑输出（逐类别明细只在最终评估时打印）。"""
    sim = res.get("similarity", {}) or {}
    ana = res.get("analogy", {}) or {}
    parts: List[str] = []
    if sim:
        parts.append(f"相似度 rho={_fmt(sim.get('spearman'))}"
                     f"（覆盖 {100 * sim.get('coverage', 0):.0f}%）")
    if ana:
        parts.append(f"类比 acc={_fmt(ana.get('accuracy'))}"
                     f"（{ana.get('n_evaluated', 0)}/{ana.get('n_questions', 0)} 题）")
    logger.info("[epoch %d 评估] %s", epoch + 1, "  |  ".join(parts) or "无可用评测集")

    grp = ana.get("by_group") or {}
    if grp:   # 只有数据集自带官方分组时才有这一项（内置小集没有）
        logger.info("            语义=%.4f  句法=%.4f",
                    grp.get("semantic", {}).get("acc", float("nan")),
                    grp.get("syntactic", {}).get("acc", float("nan")))

    if stopper is not None:
        logger.info("            监控 %s = %s   最优: %s   未提升 %d/%d",
                    canonical_metric(monitor), _fmt(metric),
                    stopper.summary(), stopper.bad_epochs, stopper.patience)


def run_epoch_eval(model, vocab: Vocab, out_dir: Path, epoch: int,
                   cfg: Dict[str, Any], monitor: str,
                   logger=None, writer=None, global_step: int = 0,
                   verbose: bool = False) -> Tuple[Dict, Optional[float]]:
    """跑一次完整评估，导出当前词向量并落盘结果 JSON。

    中途评估与最终评估用的是**同一套评测集**（都由配置里的 eval.similarity_file /
    eval.analogy_file 决定），这样 epoch 之间的数字才可比、也才能真正拿来做早停。
    verbose=False 时抑制逐类别明细，由调用方打印紧凑摘要。
    """
    from .evaluate import evaluate_from_npz

    ev = cfg.get("eval", {}) or {}
    mat = model.export_vectors(cfg["model"].get("export", "input")).float().cpu().numpy()
    # 中间快照只留 .npz：.txt 在大词表下每份上百 MB，每轮存一份太浪费
    npz_path, _ = save_vectors(out_dir / f"epoch{epoch:03d}_vectors", vocab.itos, mat,
                               write_txt=False)

    res = evaluate_from_npz(
        npz_path,
        ev.get("similarity_file"),
        ev.get("analogy_file"),
        int(ev.get("topk", 10)),
        logger=logger if verbose else None,
        fetch_eval=bool(ev.get("fetch", True)),
        max_analogy=ev.get("max_analogy"),
    )
    save_json(res, out_dir / f"eval_epoch{epoch:03d}.json")

    metric = extract_metric(res, monitor)
    if writer is not None:
        sim = res.get("similarity", {}) or {}
        ana = res.get("analogy", {}) or {}
        for tag, val in (("eval/similarity_spearman", sim.get("spearman")),
                         ("eval/analogy_accuracy", ana.get("accuracy")),
                         ("eval/monitor", metric)):
            if is_finite(val):
                writer.add_scalar(tag, float(val), global_step)
    return res, metric


def _save_best(model, vocab: Vocab, out_dir: Path, epoch: int,
               metric: Optional[float], monitor: str, cfg: Dict[str, Any]) -> None:
    """在监控指标变好时保存最优权重与最优词向量。"""
    torch.save(model.state_dict(), out_dir / "best.pt")
    mat = model.export_vectors(cfg["model"].get("export", "input")).float().cpu().numpy()
    save_vectors(out_dir / "best_vectors", vocab.itos, mat)
    save_json({"epoch": epoch, "epoch_display": epoch + 1, "metric": metric,
               "monitor": canonical_metric(monitor)},
              out_dir / "best_meta.json")


def prefetch_eval_sets(cfg: Dict[str, Any], logger) -> None:
    """训练开始前把评测集准备好（免得训练跑完才发现下不动），并打印将用哪些集。"""
    from . import eval_data

    ev = cfg.get("eval", {}) or {}
    fetch_on = bool(ev.get("fetch", True))
    for kind, spec, loader in (("相似度", ev.get("similarity_file"), eval_data.load_similarity),
                               ("类比  ", ev.get("analogy_file"), eval_data.load_analogy)):
        if not spec:
            continue
        label, items = loader(spec, fetch_on, quiet=False)
        logger.info("%s评测集: %s（%d 项）", kind, label, len(items))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_overrides(items) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set 需要 key=value 形式，收到: {item}")
        k, v = item.split("=", 1)
        try:
            import yaml

            out[k.strip()] = yaml.safe_load(v)
        except Exception:  # noqa: BLE001
            out[k.strip()] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Word2Vec 训练")
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--set", dest="overrides", action="append",
                   help="覆盖配置，例如 --set train.device=cpu --set model.dim=100")
    p.add_argument("--resume", default=None, help="从检查点续训，例如 runs/full/last.pt")
    p.add_argument("--no-eval", action="store_true",
                   help="不做任何评估（含中途评估，并自动关闭早停）")
    p.add_argument("--viz", action="store_true", help="训练后自动出可视化图")
    a = p.parse_args()

    cfg = load_config(a.config, parse_overrides(a.overrides))
    if a.no_eval:
        # --no-eval 是「这一轮不要评估」，中途评估和早停（依赖评估指标）也一并关掉，
        # 否则会出现「说了不评估却每轮都在评估、还提前停了」这种意外
        cfg.setdefault("eval", {})["every_epochs"] = 0
        cfg.setdefault("early_stopping", {})["enabled"] = False

    try:
        train(cfg, resume=a.resume, do_eval=not a.no_eval, do_viz=a.viz)
    except RuntimeError as exc:
        # 例如「请求 CUDA 但卡未开启」——上面的提醒已经打印过，这里只给结论，不甩堆栈
        print(f"\n[训练中止] {exc}\n", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
