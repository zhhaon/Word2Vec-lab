"""词向量的存取与常用运算（推理 / 评估 / 可视化共用的底层）。

文件格式：
    vectors.npz   words (V,) 字符串数组 + vectors (V, D) float32
    vectors.txt   word2vec 文本格式，首行 "V D"，可被 gensim 直接读取
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# 存取
# --------------------------------------------------------------------------
def save_vectors(prefix, words: Sequence[str], vectors: np.ndarray,
                 skip_special: bool = True) -> Tuple[Path, Path]:
    """保存为 .npz + .txt 两种格式。"""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    words = np.asarray(words, dtype=object)
    vectors = np.asarray(vectors, dtype=np.float32)

    npz_path = prefix.with_suffix(".npz")
    np.savez_compressed(npz_path, words=words, vectors=vectors)

    txt_path = prefix.with_suffix(".txt")
    keep = np.array([not (w.startswith("<") and w.endswith(">")) for w in words])
    with open(txt_path, "w", encoding="utf-8") as f:
        idx = np.nonzero(keep)[0]
        f.write(f"{len(idx)} {vectors.shape[1]}\n")
        for i in idx:
            f.write(words[i] + " " + " ".join(f"{v:.5f}" for v in vectors[i]) + "\n")
    return npz_path, txt_path


def load_vectors(path) -> "EmbeddingMatrix":
    """读取 .npz 或 .txt 词向量文件。"""
    path = Path(path)
    if not path.exists() and path.suffix == "":
        for cand in (path.with_suffix(".npz"), path.with_suffix(".txt")):
            if cand.exists():
                path = cand
                break
    if not path.exists():
        raise FileNotFoundError(f"找不到词向量文件: {path}")

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        words = [str(w) for w in data["words"]]
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    else:
        words, rows = [], []
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().split()
            dim = int(header[1]) if len(header) == 2 else None
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) < 2:
                    continue
                words.append(" ".join(parts[:-dim]) if dim and len(parts) - 1 > dim else parts[0])
                rows.append([float(x) for x in parts[-dim:] if dim])
        vectors = np.asarray(rows, dtype=np.float32)
    return EmbeddingMatrix(words, vectors)


# --------------------------------------------------------------------------
# 向量矩阵
# --------------------------------------------------------------------------
class EmbeddingMatrix:
    """带 L2 归一化的词向量查询表，提供近邻 / 相似度 / 类比。"""

    def __init__(self, words: Sequence[str], vectors: np.ndarray, normalize: bool = True) -> None:
        self.words: List[str] = list(words)
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.words)}
        mat = np.asarray(vectors, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.maximum(norms, 1e-9)
        self.matrix = mat
        self.normalized = normalize

    def __len__(self) -> int:
        return len(self.words)

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    def index(self, word: str) -> int:
        if word not in self.stoi:
            raise KeyError(word)
        return self.stoi[word]

    def vector(self, word: str) -> np.ndarray:
        return self.matrix[self.index(word)]

    def suggest(self, word: str, k: int = 5) -> List[str]:
        """拼写纠错：找出与输入最接近的词表项（按编辑距离）。"""
        import difflib

        return difflib.get_close_matches(word.lower(), self.words, n=k, cutoff=0.6)

    # ---------------- 近邻 ----------------
    def nearest(self, vec: np.ndarray, topk: int = 10,
                exclude: Iterable[str] = ()) -> List[Tuple[str, float]]:
        if self.normalized:
            q = vec / max(float(np.linalg.norm(vec)), 1e-9)
        else:
            q = vec
            norms = np.linalg.norm(self.matrix, axis=1)
            return self._topk(self.matrix @ q / np.maximum(norms * np.linalg.norm(q), 1e-9), topk, exclude)
        return self._topk(self.matrix @ q, topk, exclude)

    def _topk(self, scores: np.ndarray, topk: int,
              exclude: Iterable[str]) -> List[Tuple[str, float]]:
        exclude_idx = {self.stoi[w] for w in exclude if w in self.stoi}
        k = min(len(scores), topk + len(exclude_idx))
        idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]
        out: List[Tuple[str, float]] = []
        for i in idx:
            if int(i) in exclude_idx:
                continue
            out.append((self.words[int(i)], float(scores[int(i)])))
            if len(out) >= topk:
                break
        return out

    def most_similar(self, word: str, topk: int = 10,
                     exclude: Iterable[str] = ()) -> List[Tuple[str, float]]:
        excl = set(exclude) | {word}
        return self.nearest(self.vector(word), topk=topk, exclude=excl)

    # ---------------- 相似度 ----------------
    def similarity(self, w1: str, w2: str) -> float:
        a, b = self.vector(w1), self.vector(w2)
        if self.normalized:
            return float(a @ b)
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    # ---------------- 类比 ----------------
    def analogy(self, a: str, b: str, c: str, topk: int = 5) -> List[Tuple[str, float]]:
        """a 之于 b，相当于 c 之于 ?   ->   vec(b) - vec(a) + vec(c)"""
        vec = self.vector(b) - self.vector(a) + self.vector(c)
        return self.nearest(vec, topk=topk, exclude={a, b, c})

    def analogy_top1(self, a: str, b: str, c: str) -> Optional[str]:
        res = self.analogy(a, b, c, topk=1)
        return res[0][0] if res else None

    def arithmetic(self, positive: Sequence[str], negative: Sequence[str] = (),
                   topk: int = 10) -> List[Tuple[str, float]]:
        """通用向量算术：sum(positive) - sum(negative)。"""
        vec = np.zeros(self.dim, dtype=np.float32)
        for w in positive:
            vec = vec + self.vector(w)
        for w in negative:
            vec = vec - self.vector(w)
        excl = set(positive) | set(negative)
        return self.nearest(vec, topk=topk, exclude=excl)

    # ---------------- 子集 ----------------
    def subset(self, words: Sequence[str]) -> "EmbeddingMatrix":
        idx = [self.stoi[w] for w in words if w in self.stoi]
        return EmbeddingMatrix([self.words[i] for i in idx], self.matrix[idx], normalize=False)
