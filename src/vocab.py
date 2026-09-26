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

# 特殊符号：不携带语义，任何情况下都不允许被抽成负样本。
# 它们固定占住最小的下标 [0, NUM_SPECIALS)，所以「是否为特殊符号」可以直接用 id < NUM_SPECIALS 判断。
SPECIAL_IDS = (UNK_IDX, PAD_IDX)
SPECIAL_SET = frozenset(SPECIAL_IDS)
NUM_SPECIALS = len(SPECIAL_TOKENS)


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
        """构建 unigram^0.75 的槽位表：返回 int32 数组，可直接用随机下标取词 id。

        频次越高的词占据越多的槽位；用「查表 + 随机下标」代替 np.random.choice(p=...)，
        避免每次采样都做 O(V) 扫描，速度快几个数量级。

        <unk> / <pad> 的词频被清零，因此不会占据任何槽位 —— 这是排除它们的第一道防线。
        """
        counts = self.counts.astype(np.float64).copy()
        # ① 把 <unk> / <pad> 的词频清零 -> p^0.75 = 0 -> 分不到任何槽位
        counts[: len(SPECIAL_TOKENS)] = 0.0

        if len(self) <= len(SPECIAL_TOKENS):
            raise ValueError("词表里没有任何实词，无法构建负采样表")
        if not (counts > 0).any():
            raise ValueError("所有实词词频都是 0，无法构建负采样表")

        if table_size is None:
            table_size = int(min(50_000_000, max(1_000_000, 20 * len(self))))

        probs = np.power(counts, 0.75)
        probs /= probs.sum()

        real_slots = np.floor(probs * table_size).astype(np.int64)
        # ② 把整除损失的余数补给最高频的实词；下标 2 起才是实词
        diff = table_size - int(real_slots.sum())
        if diff != 0:
            real_slots[2 + int(np.argmax(counts[2:]))] += diff

        table = np.repeat(np.arange(len(real_slots), dtype=np.int32), real_slots)

        # ③ 兜底断言：表里绝不能出现 <unk> / <pad>
        if np.isin(table, SPECIAL_IDS).any():
            raise AssertionError("负采样表里出现了 <unk> / <pad>，请检查词表构建逻辑")
        return table


class NegativeSampler:
    """从负采样表里批量取负样本。

    两条硬约束：
      1. 负样本不能是该样本的 **positive target**（也就是 label=1 的那个词）。
         否则等于手把手教模型「真正的目标不是目标」，是典型的假负样本；
      2. 负样本不能是 <unk> / <pad>。它们不携带语义，
         被抽中只是白送噪声，还会让 <unk> 的向量被大量无意义梯度污染。

    采样表本身已经排除了特殊符号，但 :meth:`sample_excluding` 仍会显式再过滤一遍，
    这样即使以后换了别的采样表实现，这两条约束也不会被悄悄破坏。
    """

    def __init__(self, vocab: Vocab, table_size: Optional[int] = None, seed: int = 0) -> None:
        self.table = vocab.negative_sampling_table(table_size)
        self.rng = np.random.default_rng(seed)
        self.vocab_size = len(vocab)

        # 统计量：用来确认「排除正目标」这条规则真的生效了
        self.n_sampled = 0        # 累计抽出的负样本个数
        self.n_resampled = 0      # 其中因为撞上正目标 / 特殊符号而被重抽的个数

        if np.isin(self.table, SPECIAL_IDS).any():
            raise AssertionError("负采样表里出现了 <unk> / <pad>")

    # ---------------- 基础采样 ----------------
    def sample(self, shape) -> np.ndarray:
        idx = self.rng.integers(0, len(self.table), size=shape, dtype=np.int64)
        return self.table[idx]

    def sample_array(self, n: int) -> np.ndarray:
        return self.sample((n,))

    # ---------------- 带约束的采样 ----------------
    def sample_excluding(self, shape, targets, extra=None, max_tries: int = 8) -> np.ndarray:
        """采负样本，并保证每一行都不含「本行禁止出现的词」。

        参数
        ----
        shape   : (B, K)，B 是 batch 大小，K 是每个正样本配的负样本数
        targets : (B,) 每行的 positive target id（模型里做正例的那个词）
        extra   : (B,) 或 (B, M)，可选。同属假负样本的额外禁用词
                  （Skip-gram 传中心词，CBOW 传整条上下文），可为 None
        max_tries : 拒绝采样轮数上限。词表正常时一轮就结束
                    （冲突概率约 K/V；text8 的 V≈7.1 万、K=10 时约 1.4e-4）

        返回 (B, K) int32 数组。
        """
        shape = tuple(int(s) for s in shape)
        b = shape[0]
        # 统一用 int32，与采样表 dtype 保持一致。
        # 否则 numpy 会在每次比较时把 int32 隐式升级成 int64 再拷贝一份，白白慢一倍。
        targets = np.asarray(targets, dtype=np.int32).reshape(b)
        if extra is not None:
            extra = np.asarray(extra, dtype=np.int32)
            if extra.ndim == 1:
                extra = extra.reshape(b, 1)

        neg = self.sample(shape)
        forbidden = self._forbidden_mask(neg, targets, extra)

        tries = 0
        while forbidden.any() and tries < max_tries:
            n_bad = int(forbidden.sum())
            neg[forbidden] = self.sample_array(n_bad)
            self.n_resampled += n_bad
            forbidden = self._forbidden_mask(neg, targets, extra)
            tries += 1

        if forbidden.any():
            # 词表极小时才可能走到这里：确定性地挑一个合法 id
            self._force_valid(neg, forbidden, targets, extra)

        self.n_sampled += neg.size
        return neg

    @staticmethod
    def _forbidden_mask(neg: np.ndarray, targets: np.ndarray,
                        extra: Optional[np.ndarray]) -> np.ndarray:
        """标出哪些位置是非法的：等于正目标、是 <unk>/<pad>、或落在 extra 里。"""
        # ① 等于本行的正目标
        mask = neg == targets[:, None]
        # ② 特殊符号：它们固定占住 [0, NUM_SPECIALS)，一次比较即可
        mask |= neg < NUM_SPECIALS
        # ③ extra 里逐个比较。按列循环而不是做 (B, K, M) 三维广播，
        #    实测在 B=8192/K=10/M=10 时快 3 倍以上（少了一个大临时数组和一次归约）
        if extra is not None:
            for m in range(extra.shape[1]):
                mask |= neg == extra[:, m:m + 1]
        return mask

    def _force_valid(self, neg: np.ndarray, forbidden: np.ndarray,
                     targets: np.ndarray, extra: Optional[np.ndarray]) -> None:
        rows, cols = np.nonzero(forbidden)
        for r, c in zip(rows, cols):
            banned = {int(targets[r])} | SPECIAL_SET
            if extra is not None:
                banned.update(int(x) for x in extra[r])
            cand = 2
            while cand < self.vocab_size and cand in banned:
                cand += 1
            if cand < self.vocab_size:
                neg[r, c] = cand

    # ---------------- 统计 ----------------
    @property
    def resample_rate(self) -> float:
        return self.n_resampled / self.n_sampled if self.n_sampled else 0.0
