"""第 6 步（量化部分）：词向量评估。

两套公开可比的指标：
    1) 词相似度：与人工打分做 Spearman 相关（衡量「语义距离」是否合理）
    2) 词类比  ：3CosAdd 准确率（a:b :: c:?），衡量「线性结构」是否学到

评测集可以是下面任意一种（`src/eval_data.py` 里有完整登记表）：
    内置小评测集（离线、秒级）      data/eval/similarity_pairs.txt / analogy_questions.txt
    标准公开数据集（按需下载缓存）  wordsim353（353 行）/ google-analogy（19544 题 / 14 类）
                                   另有 wordsim353-sim / wordsim353-rel / men / simlex999

用法：
    python -m src.evaluate --vectors runs/full/vectors.npz \
        --similarity wordsim353 --analogy google-analogy
    python -m src.evaluate --vectors runs/tiny/vectors.npz \
        --similarity data/eval/similarity_pairs.txt
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import eval_data
from .utils import human_int, resolve_path, save_json
from .vectors import EmbeddingMatrix, load_vectors


# --------------------------------------------------------------------------
# 读取评测文件
# --------------------------------------------------------------------------
def read_similarity_pairs(path) -> List[Tuple[str, str, float]]:
    """从本地文件读词相似度对（兼容 tab / 空格 / CSV 三种写法）。"""
    p = resolve_path(path)
    if p is None or not p.exists():
        raise FileNotFoundError(f"找不到相似度评测文件: {path}")
    return eval_data.parse_similarity_text(p.read_text(encoding="utf-8", errors="ignore"))


def read_analogy_questions(path) -> List[Tuple[str, str, str, str]]:
    """从本地文件读类比题（兼容 Google Analogy 的 ': 分组' 段头）。"""
    p = resolve_path(path)
    if p is None or not p.exists():
        raise FileNotFoundError(f"找不到类比评测文件: {path}")
    return [(a, b, c, d) for a, b, c, d, _ in
            eval_data.parse_analogy_text(p.read_text(encoding="utf-8", errors="ignore"))]


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（不依赖 scipy 的实现，避免额外依赖）。"""
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def eval_similarity(emb: EmbeddingMatrix, pairs, logger=None, label: str = "") -> Dict:
    gold, pred, used, missing = [], [], [], []
    for w1, w2, score in pairs:
        if w1 in emb and w2 in emb:
            gold.append(score)
            pred.append(emb.similarity(w1, w2))
            used.append((w1, w2, score, pred[-1]))
        else:
            missing.append((w1, w2))

    dup = [p for p, n in Counter((a, b) for a, b, _ in pairs).items() if n > 1]
    result: Dict = {
        "dataset": label,
        "n_pairs": len(pairs),
        "n_unique_pairs": len({(a, b) for a, b, _ in pairs}),
        "n_used": len(used),
        "coverage": len(used) / max(1, len(pairs)),
        "n_oov": len(missing),
        "duplicated_pairs": [f"{a}-{b}" for a, b in dup[:10]],
    }
    if len(used) < 2:
        result["spearman"] = float("nan")
        if logger:
            logger.warning("相似度评估(%s): 只有 %d 对可评估，无法计算相关系数",
                           label or "?", len(used))
        return result

    gold_arr = np.asarray(gold, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    rho = spearman(gold_arr, pred_arr)
    result["spearman"] = rho
    result["pearson"] = float(np.corrcoef(gold_arr, pred_arr)[0, 1]) \
        if gold_arr.std() > 0 and pred_arr.std() > 0 else float("nan")

    # 最准 / 最离谱的样本，便于人工检查
    used_sorted = sorted(used, key=lambda t: -(t[2]))
    result["top_pairs"] = [
        {"words": f"{w1}-{w2}", "gold": g, "pred": round(p, 3)}
        for w1, w2, g, p in used_sorted[:5]
    ]
    result["worst_pairs"] = [
        {"words": f"{w1}-{w2}", "gold": g, "pred": round(p, 3)}
        for w1, w2, g, p in used_sorted[-5:]
    ]
    result["missing"] = [f"{a}/{b}" for a, b in missing[:20]]

    if logger:
        logger.info("相似度评估 [%s]: Spearman rho = %.4f   (%d/%d 对可评估，覆盖率 %.1f%%)",
                    label or "?", rho, len(used), len(pairs), 100 * result["coverage"])
        if result["n_unique_pairs"] != result["n_pairs"]:
            logger.info("    注: 该数据集含重复词对 %s，共 %d 行 / %d 个唯一词对",
                        ", ".join(result["duplicated_pairs"]),
                        result["n_pairs"], result["n_unique_pairs"])
    return result


def eval_analogy(emb: EmbeddingMatrix, questions, topk: int = 1, logger=None,
                 label: str = "", chunk_mb: int = 128,
                 max_questions: Optional[int] = None) -> Dict:
    """3CosAdd 类比评估：vec(b) - vec(a) + vec(c) 的最近邻是否为 d。

    实现要点：
      * 先把所有题的 query 向量一次性算出来，再分块做矩阵乘。
        逐题算的话，Google Analogy（19544 题）× 7.1 万词表要十几分钟，
        分块 GEMM 让 BLAS 吃满，只要数十秒；
      * 分块大小按「得分矩阵不超过 chunk_mb」定，避免 19544×250000 的巨型中间矩阵；
      * 逐条排除 a/b/c 三个词，保证答案不会是自己。
    """
    if max_questions:
        questions = questions[:max_questions]
    if not questions:
        return {"dataset": label, "n_questions": 0, "accuracy": float("nan")}

    M = emb.matrix
    V = len(emb)
    stoi = emb.stoi
    n = len(questions)

    def _idx(pos: int) -> np.ndarray:
        return np.fromiter((stoi.get(q[pos], -1) for q in questions),
                           dtype=np.int64, count=n)

    a_i, b_i, c_i, d_i = (_idx(0), _idx(1), _idx(2), _idx(3))
    # a/b/c 有任意一个不在词表就没法评估；d 不在词表则必然答错（保留在分母里）
    valid = (a_i >= 0) & (b_i >= 0) & (c_i >= 0)
    n_oov_probe = int((~valid).sum())

    vq = [q for q, ok in zip(questions, valid) if ok]
    va, vb, vc, vd = a_i[valid], b_i[valid], c_i[valid], d_i[valid]
    N = len(vq)

    result: Dict = {
        "dataset": label,
        "n_questions": n,
        "n_evaluated": N,
        "n_oov_probe": n_oov_probe,          # a/b/c 缺词，无法评估
        "coverage": N / max(1, n),
    }
    if N == 0:
        result["accuracy"] = float("nan")
        if logger:
            logger.warning("类比评估 [%s]: 没有一道题的 a/b/c 都在词表里", label or "?")
        return result

    # ---------- 一次性算出所有 query 向量 ----------
    Q = M[vb] - M[va] + M[vc]                     # (N, D) float32

    # ---------- 分块 GEMM + top-k ----------
    k = min(max(int(topk), 1), V)
    step = max(16, min(4096, (chunk_mb << 20) // max(1, V * 4)))
    pred = np.empty(N, dtype=np.int64)

    for s in range(0, N, step):
        e = min(s + step, N)
        scores = Q[s:e] @ M.T                     # (b, V)
        rows = np.arange(e - s)
        scores[rows, va[s:e]] = -np.inf           # 排除 a/b/c
        scores[rows, vb[s:e]] = -np.inf
        scores[rows, vc[s:e]] = -np.inf
        part = np.argpartition(scores, V - k, axis=1)[:, V - k:]
        vals = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-vals, axis=1)
        pred[s:e] = np.take_along_axis(part, order, axis=1)[:, 0]

    # ---------- 统计 ----------
    words = np.asarray(emb.words, dtype=object)
    pred_words = words[pred]
    gold_words = np.asarray([q[3] for q in vq], dtype=object)
    correct = pred_words == gold_words

    n_answer_oov = int((vd < 0).sum())
    n_correct = int(correct.sum())
    result.update({
        "accuracy": n_correct / N,
        "n_correct": n_correct,
        "accuracy_total": n_correct / n,          # 缺词的题算错，最严格的口径
        "n_answer_oov": n_answer_oov,
    })

    # ---------- 按类别 / 语义-句法分组 ----------
    # Google Analogy 带 ": 分组名" 段头；内置小评测集没有，只能按已知词表推断。
    has_official_cats = any(len(q) > 4 and q[4] for q in vq)
    cats = [q[4] if len(q) > 4 and q[4] else _guess_category(q[0], q[1], q[2], q[3])
            for q in vq]
    by_cat: Dict[str, List[bool]] = {}
    for cat, ok in zip(cats, correct):
        by_cat.setdefault(cat, []).append(bool(ok))
    result["by_category"] = {
        cat: {"n": len(v), "acc": float(np.mean(v)), "correct": int(np.sum(v))}
        for cat, v in sorted(by_cat.items())
    }
    result["categories_official"] = has_official_cats

    # 语义/句法二分只对官方类别有意义：内置集的类别是猜出来的，
    # 硬套这二分法会把 capital、gender 全算进「句法」，得出误导性的结论。
    if has_official_cats:
        groups: Dict[str, List[bool]] = {"semantic": [], "syntactic": []}
        for cat, v in by_cat.items():
            groups["semantic" if cat in eval_data.SEMANTIC_GROUPS
                   else "syntactic"].extend(v)
        result["by_group"] = {
            g: {"n": len(v), "acc": float(np.mean(v)) if v else float("nan")}
            for g, v in groups.items() if v
        }

    items = [{"q": f"{q[0]}:{q[1]} :: {q[2]}:{p}", "gold": g}
             for q, p, g in zip(vq, pred_words, gold_words)]
    result["examples_correct"] = [it for it, ok in zip(items, correct) if ok][:5]
    result["examples_wrong"] = [it for it, ok in zip(items, correct) if not ok][:5]

    if logger:
        logger.info("类比评估 [%s]: 准确率 = %.4f   (%d/%d 题可评估，%d 题因 a/b/c 缺词跳过)",
                    label or "?", result["accuracy"], N, n, n_oov_probe)
        logger.info("    严格口径（缺词的题算错）= %.4f   答案本身缺词的题: %d",
                    result["accuracy_total"], n_answer_oov)
        for g, v in (result.get("by_group") or {}).items():
            logger.info("    %-10s n=%6d  acc=%.4f", g, v["n"], v["acc"])
        by_cat_sorted = sorted(result["by_category"].items())
        if len(by_cat_sorted) <= 20:
            for cat, v in by_cat_sorted:
                logger.info("      %-26s n=%5d  acc=%.4f", cat, v["n"], v["acc"])
    return result


def _guess_category(a: str, b: str, c: str, d: str) -> str:
    """按已知词表粗分组，纯粹为了让报告更好读。"""
    groups = {
        "gender": {"king", "queen", "man", "woman", "boy", "girl", "father",
                   "mother", "son", "daughter", "brother", "sister", "uncle",
                   "aunt", "prince", "princess", "husband", "wife", "actor",
                   "actress", "grandfather", "grandmother", "waiter", "waitress"},
        "capital": {"paris", "france", "berlin", "germany", "rome", "italy",
                    "madrid", "spain", "tokyo", "japan", "moscow", "russia",
                    "beijing", "china", "london", "england", "cairo", "egypt",
                    "athens", "greece", "lisbon", "portugal"},
        "language": {"french", "german", "italian", "spanish", "japanese",
                     "russian", "chinese", "arabic", "greek"},
        "animal": {"dog", "puppy", "cat", "kitten"},
        "comparative": {"big", "bigger", "small", "smaller", "tall", "taller",
                        "short", "shorter", "old", "older", "new", "newer",
                        "long", "longer", "fast", "faster", "slow", "slower",
                        "warm", "warmer", "cold", "colder", "easy", "easier",
                        "hard", "harder", "close", "closer", "open", "opener"},
        "opposite": {"good", "bad", "hot", "light", "dark", "begin", "end"},
    }
    words = {a, b, c, d}
    best, best_overlap = "other", 0
    for name, vocab in groups.items():
        overlap = len(words & vocab)
        if overlap > best_overlap:
            best, best_overlap = name, overlap
    return best


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def evaluate_from_npz(vectors_path, similarity_file: Optional[str],
                      analogy_file: Optional[str], topk: int = 10,
                      logger=None, fetch_eval: bool = True,
                      max_analogy: Optional[int] = None) -> Dict:
    """统一评估入口。

    similarity_file / analogy_file 既可以是注册数据集名（wordsim353、google-analogy …），
    也可以是本地文件路径。注册名会按需下载并缓存，下载失败自动回退到内置小评测集。
    """
    emb = load_vectors(vectors_path)
    if logger:
        logger.info("评估词向量: %s  (%s 词, %d 维)",
                    Path(vectors_path).name, human_int(len(emb)), emb.dim)
    out: Dict = {"n_words": len(emb), "dim": emb.dim, "vectors": str(vectors_path)}

    if similarity_file:
        label, pairs = eval_data.load_similarity(similarity_file, fetch_eval,
                                                 quiet=logger is None)
        if pairs:
            if logger:
                logger.info("相似度评测集: %s（%d 行）", label, len(pairs))
            out["similarity"] = eval_similarity(emb, pairs, logger, label)
        elif logger:
            logger.warning("没有可用的相似度评测集: %s", similarity_file)

    if analogy_file:
        label, questions = eval_data.load_analogy(analogy_file, fetch_eval,
                                                  quiet=logger is None)
        if questions:
            if logger:
                logger.info("类比评测集: %s（%d 题）", label, len(questions))
            out["analogy"] = eval_analogy(emb, questions, topk, logger, label,
                                          max_questions=max_analogy)
        elif logger:
            logger.warning("没有可用的类比评测集: %s", analogy_file)
    return out


def main() -> None:
    from .utils import force_utf8_stdout, get_logger

    force_utf8_stdout()
    p = argparse.ArgumentParser(description="词向量评估")
    p.add_argument("--vectors", required=True, help="vectors.npz 或 vectors.txt")
    p.add_argument("--similarity", "--similarity-file", dest="similarity",
                   default="data/eval/similarity_pairs.txt",
                   help="数据集名（wordsim353 / men / simlex999 …）或文件路径")
    p.add_argument("--analogy", "--analogy-file", dest="analogy",
                   default="data/eval/analogy_questions.txt",
                   help="数据集名（google-analogy）或文件路径")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--no-fetch", action="store_true", help="不联网下载，只用缓存/本地文件")
    p.add_argument("--max-analogy", type=int, default=None,
                   help="只用前 N 道类比题（快速自检用）")
    p.add_argument("--out", default=None, help="结果 JSON 保存路径")
    p.add_argument("--list-datasets", action="store_true", help="列出可用数据集")
    a = p.parse_args()

    if a.list_datasets:
        for name, spec in eval_data.REGISTRY.items():
            print(f"  {name:16s} [{spec['kind']:10s}] {spec['desc']}")
        return

    logger = get_logger("word2vec")
    res = evaluate_from_npz(a.vectors, a.similarity, a.analogy, a.topk, logger,
                            fetch_eval=not a.no_fetch, max_analogy=a.max_analogy)
    if a.out:
        save_json(res, resolve_path(a.out))
        logger.info("结果已保存: %s", a.out)


if __name__ == "__main__":
    main()
