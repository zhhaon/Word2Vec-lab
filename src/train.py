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
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import PairStream
from .model import Word2VecNeg, build_optimizer, estimate_parameters, lr_at_step
from .preprocess import load_processed, preprocess as run_preprocess
from .utils import (Timer, ensure_dir, format_hms, get_logger, human_int,
                    load_config, resolve_path, set_seed)
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
            for item in batches:
                if self._stop.is_set():
                    break
                epoch, center, context = item
                self.q.put((epoch, self._to_tensor(center), self._to_tensor(context)))
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

    def batch_iter() -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        for ep in range(start_epoch, epochs):
            for c, x in stream.iter_batches(batch_size=batch_size, epoch=ep):
                yield ep, c, x

    prefetcher = Prefetcher(batch_iter(), depth=prefetch, device=device)
    timer = Timer()
    running_loss, run_steps, seen_pairs = 0.0, 0, 0
    stopped_early = False
    last_epoch = start_epoch - 1

    logger.info("-" * 74)
    logger.info("开始训练（max_minutes=%s，到点会自动停止并保存）",
                max_minutes if max_minutes > 0 else "无限制")
    logger.info("-" * 74)

    try:
        for epoch, center, context in prefetcher:
            # 每个 epoch 交界处，按需做一次中途评估（看清指标随训练怎么变化）
            if epoch != last_epoch:
                last_epoch = epoch
                if eval_every > 0 and epoch > 0 and epoch % eval_every == 0:
                    _quick_eval(model, vocab, out_dir, epoch, cfg, logger)

            if global_step >= total_steps:
                global_step = total_steps - 1  # 学习率调度收尾
            lr = lr_at_step(global_step, total_steps, base_lr, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            bs = center.size(0)
            negatives = neg_sampler.sample_array(bs * n_neg) if n_neg > 0 else None
            if negatives is not None:
                negatives = torch.from_numpy(negatives.reshape(bs, n_neg)).to(
                    device, non_blocking=True)

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
                logger.warning("已达到时间预算 %.0f 分钟，优雅停止并保存。", max_minutes)
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
    npz_path, txt_path = export_vectors(model, vocab, out_dir, m_cfg.get("export", "input"))
    logger.info("词向量已导出: %s / %s", npz_path.name, txt_path.name)
    logger.info("总用时 %s%s", format_hms(timer.elapsed),
                "（因时间预算提前停止）" if stopped_early else "")

    if writer:
        writer.flush()
        writer.close()

    # ---------------- 评估 / 可视化 ----------------
    if do_eval:
        try:
            from .evaluate import evaluate_from_npz

            ev = cfg.get("eval", {})
            res = evaluate_from_npz(npz_path, ev.get("similarity_file"),
                                    ev.get("analogy_file"), int(ev.get("topk", 10)),
                                    logger=logger)
            from .utils import save_json

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


def _quick_eval(model, vocab: Vocab, out_dir: Path, epoch: int,
                cfg: Dict[str, Any], logger) -> None:
    """训练途中的轻量评估：导出当前向量并算指标，方便观察收敛过程。"""
    try:
        from .evaluate import evaluate_from_npz

        ev = cfg.get("eval", {}) or {}
        npz_path, _ = save_vectors(out_dir / f"epoch{epoch:03d}_vectors",
                                   vocab.itos,
                                   model.export_vectors(
                                       cfg["model"].get("export", "input")
                                   ).float().cpu().numpy())
        res = evaluate_from_npz(npz_path, ev.get("similarity_file"),
                                ev.get("analogy_file"), int(ev.get("topk", 10)))
        sim = res.get("similarity", {}).get("spearman", float("nan"))
        ana = res.get("analogy", {}).get("accuracy", float("nan"))
        logger.info("  [epoch %d 中途评估] 相似度 rho=%.4f  类比 acc=%.4f", epoch, sim, ana)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  中途评估失败: %s", exc)


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
    p.add_argument("--no-eval", action="store_true", help="训练后不做评估")
    p.add_argument("--viz", action="store_true", help="训练后自动出可视化图")
    a = p.parse_args()

    cfg = load_config(a.config, parse_overrides(a.overrides))
    try:
        train(cfg, resume=a.resume, do_eval=not a.no_eval, do_viz=a.viz)
    except RuntimeError as exc:
        # 例如「请求 CUDA 但卡未开启」——上面的提醒已经打印过，这里只给结论，不甩堆栈
        print(f"\n[训练中止] {exc}\n", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
