"""词表：构建、编解码、以及负采样分布表。

词表约定（固定下标，全流程一致）：
    0 -> <unk>   （未登录词 / 低频词，且不作为负样本）
    1 -> <pad>   （仅 CBOW 上下文补齐使用，且不作为负样本）
    其余按下标 2.. 按词频降序排列。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

UNK = "<unk>"
PAD = "<pad>"
SPECIAL_TOKENS = [UNK, PAD]
UNK_IDX = 0
PAD_IDX = 1


class Vocab:
    def __init__(self, itos: Sequence[str], counts: Sequence[int]) -> None:
        assert len(itos) == len(counts), "itos 与 counts 长度必须一致"
        self.itos: List[str] = list(itos)
        self.counts: np.ndarray = np.asarray(counts, dtype=np.int64)
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    # ---------------- 构建 ----------------
    @classmethod
    def from_counter(
        cls,
        counter: Counter,
        min_count: int = 5,
        max_vocab: Optional[int] = None,
    ) -> "Vocab":
        """按词频降序构建词表；低于 min_count 的词统一并入 <unk>。"""
        items = [(w, c) for w, c in counter.items() if c >= min_count]
        items.sort(key=lambda x: (-x[1], x[0]))

        if max_vocab is not None and len(items) > max_vocab:
            dropped = items[max_vocab:]
            items = items[:max_vocab]
        else:
            dropped = []

        itos: List[str] = list(SPECIAL_TOKENS)
        counts: List[int] = [0, 0]
        for w, c in items:
            itos.append(w)
            counts.append(int(c))
        # <unk> 计数 = 所有被丢弃词的频次之和，便于统计覆盖率
        counts[UNK_IDX] = int(sum(c for _, c in dropped))

        # <unk> 计数为 0 时给个占位，避免除零
        if counts[UNK_IDX] == 0:
            counts[UNK_IDX] = 1
        return cls(itos, counts)

    # ---------------- 基础属性 ----------------
    def __len__(self) -> int:
        return len(self.itos)

    @property
    def size(self) -> int:
        return len(self.itos)

    def __contains__(self, w: str) -> bool:
        return w in self.stoi

    # ---------------- 编解码 ----------------
    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        stoi = self.stoi
        return np.fromiter(
            (stoi.get(w, UNK_IDX) for w in tokens), dtype=np.int32, count=-1
        )

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.itos[int(i)] for i in ids]

    def word(self, idx: int) -> str:
        return self.itos[int(idx)]

    def index(self, word: str) -> int:
        """未登录词返回 <unk>，不报错。"""
        return self.stoi.get(word, UNK_IDX)

    # ---------------- 持久化 ----------------
    def to_dict(self) -> Dict:
        return {"itos": self.itos, "counts": self.counts.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "Vocab":
        return cls(d["itos"], d["counts"])

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---------------- 负采样 ----------------
    def negative_sampling_table(self, table_size: Optional[int] = None) -> np.ndarray:
        """构建 unigram^0.75 的别名表：返回 int32 数组，可直接随机下标取词 id。

        用「查表 + 随机下标」代替 np.random.choice(p=...)，
        避免每次采样都做 O(V) 扫描，速度快几个数量级。
        """
        counts = self.counts.astype(np.float64).copy()
        counts[UNK_IDX] = 0.0   # 特殊符号不参与负采样
        counts[PAD_IDX] = 0.0

        if table_size is None:
            table_size = int(min(50_000_000, max(1_000_000, 20 * len(self))))

        probs = np.power(counts, 0.75)
        total = probs.sum()
        if total <= 0:
            raise ValueError("语料为空或所有词频为 0，无法构建负采样表")
        probs /= total

        slots = np.floor(probs * table_size).astype(np.int64)
        diff = table_size - int(slots.sum())
        if diff != 0:
            # 余下的槽位补给最高频的实词（下标 2 通常是频率最高的词）
            slots[2:][int(np.argmax(counts[2:]))] += diff
            # 若 diff 为负需回补，极少数情况下才发生
            while slots.sum() > table_size:
                slots[2:][int(np.argmax(slots[2:]))] -= 1

        table = np.repeat(np.arange(len(slots), dtype=np.int32), slots)
        return table


class NegativeSampler:
    """从负采样表里批量取负样本。"""

    def __init__(self, vocab: Vocab, table_size: Optional[int] = None, seed: int = 0) -> None:
        self.table = vocab.negative_sampling_table(table_size)
        self.rng = np.random.default_rng(seed)
        self.vocab_size = len(vocab)

    def sample(self, shape) -> np.ndarray:
        idx = self.rng.integers(0, len(self.table), size=shape, dtype=np.int64)
        return self.table[idx]

    def sample_array(self, n: int) -> np.ndarray:
        return self.sample((n,))
