"""第 6 步（量化部分）：词向量评估。

两套公开可比的指标：
    1) 词相似度：与人工打分做 Spearman 相关（衡量「语义距离」是否合理）
    2) 词类比  ：3CosAdd 准确率（a:b :: c:?），衡量「线性结构」是否学到

用法：
    python -m src.evaluate --vectors runs/full/vectors.npz
    python -m src.evaluate --vectors runs/full/vectors.npz --analogy
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import human_int, resolve_path, save_json
from .vectors import EmbeddingMatrix, load_vectors


# --------------------------------------------------------------------------
# 读取评测文件
# --------------------------------------------------------------------------
def read_similarity_pairs(path) -> List[Tuple[str, str, float]]:
    pairs: List[Tuple[str, str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pairs.append((parts[0], parts[1], float(parts[2])))
            except ValueError:
                continue
    return pairs


def read_analogy_questions(path) -> List[Tuple[str, str, str, str]]:
    qs: List[Tuple[str, str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                qs.append((parts[0], parts[1], parts[2], parts[3]))
    return qs


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


def eval_similarity(emb: EmbeddingMatrix, pairs, logger=None) -> Dict:
    gold, pred, used, missing = [], [], [], []
    for w1, w2, score in pairs:
        if w1 in emb and w2 in emb:
            gold.append(score)
            pred.append(emb.similarity(w1, w2))
            used.append((w1, w2, score, pred[-1]))
        else:
            missing.append((w1, w2))

    result: Dict = {"n_pairs": len(pairs), "n_used": len(used),
                    "coverage": len(used) / max(1, len(pairs))}
    if len(used) < 2:
        result["spearman"] = float("nan")
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
        logger.info("相似度评估: Spearman rho = %.4f   (%d/%d 对可评估，覆盖率 %.1f%%)",
                    rho, len(used), len(pairs), 100 * result["coverage"])
    return result


def eval_analogy(emb: EmbeddingMatrix, questions, topk: int = 1, logger=None) -> Dict:
    per_category: Dict[str, List[int]] = {}
    used, missing = 0, 0
    correct_examples, wrong_examples = [], []

    for a, b, c, d in questions:
        if not all(w in emb for w in (a, b, c, d)):
            missing += 1
            continue
        used += 1
        pred = emb.analogy_top1(a, b, c)
        ok = int(pred == d)
        category = _guess_category(a, b, c, d)
        per_category.setdefault(category, []).append(ok)
        item = {"q": f"{a}:{b} :: {c}:{pred}", "gold": d}
        (correct_examples if ok else wrong_examples).append(item)

    total = sum(len(v) for v in per_category.values())
    result = {
        "n_questions": len(questions),
        "n_used": used,
        "n_missing": missing,
        "accuracy": float(np.mean([v for vs in per_category.values() for v in vs]))
        if total else float("nan"),
        "top1_accuracy": float(np.mean([v for vs in per_category.values() for v in vs]))
        if total else float("nan"),
        "by_group": {k: {"n": len(v), "acc": float(np.mean(v))}
                     for k, v in sorted(per_category.items())},
        "examples_correct": correct_examples[:5],
        "examples_wrong": wrong_examples[:5],
    }
    if logger:
        logger.info("类比评估: 准确率 = %.4f   (%d/%d 题可评估)",
                    result["accuracy"], used, len(questions))
        for k, v in result["by_group"].items():
            logger.info("    %-14s n=%2d  acc=%.2f", k, v["n"], v["acc"])
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
                      logger=None) -> Dict:
    emb = load_vectors(vectors_path)
    if logger:
        logger.info("评估词向量: %s  (%s 词, %d 维)",
                    Path(vectors_path).name, human_int(len(emb)), emb.dim)
    out: Dict = {"n_words": len(emb), "dim": emb.dim, "vectors": str(vectors_path)}

    if similarity_file:
        p = resolve_path(similarity_file)
        if p and p.exists():
            out["similarity"] = eval_similarity(emb, read_similarity_pairs(p), logger)
        elif logger:
            logger.warning("找不到相似度评测文件: %s", similarity_file)

    if analogy_file:
        p = resolve_path(analogy_file)
        if p and p.exists():
            out["analogy"] = eval_analogy(emb, read_analogy_questions(p), topk, logger)
        elif logger:
            logger.warning("找不到类比评测文件: %s", analogy_file)
    return out


def main() -> None:
    from .utils import get_logger

    p = argparse.ArgumentParser(description="词向量评估")
    p.add_argument("--vectors", required=True, help="vectors.npz 或 vectors.txt")
    p.add_argument("--similarity-file", default="data/eval/similarity_pairs.txt")
    p.add_argument("--analogy-file", default="data/eval/analogy_questions.txt")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out", default=None, help="结果 JSON 保存路径")
    a = p.parse_args()

    logger = get_logger("word2vec")
    res = evaluate_from_npz(a.vectors, a.similarity_file, a.analogy_file, a.topk, logger)
    if a.out:
        save_json(res, resolve_path(a.out))
        logger.info("结果已保存: %s", a.out)


if __name__ == "__main__":
    main()
