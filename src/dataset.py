"""第 3 步：训练样本生成。

Word2Vec 的两类样本：
    Skip-gram : 用中心词预测上下文   ->  (center, context)
    CBOW      : 用上下文预测中心词   ->  (context[], center)

本模块用**全向量化的 numpy** 生成样本对，而不是 Python 逐词循环。
原因：1700 万词的语料，一个 epoch 需要生成约 1 亿个样本对，
纯 Python 循环会成为绝对瓶颈（分钟级 -> 十几分钟），向量化后只需秒级。

同时实现论文中的两个关键技巧：
    1) 动态窗口：每个中心词的窗口半径 r ~ Uniform{1..W}，离得近的词被采到更多次；
    2) 高频词下采样 (subsampling)：P(保留) = sqrt(t/f) + t/f，削弱 "the/of" 这类词的支配地位。
"""
from __future__ import annotations

import argparse
from typing import Iterator, Optional, Tuple

import numpy as np

from .utils import human_int
from .vocab import PAD_IDX, UNK_IDX, Vocab


class PairStream:
    """向量化的 (center[s], context[s]) 样本流。

    用法：
        stream = PairStream(tokens, sent_lengths, vocab, window=5, arch="skipgram")
        for epoch in range(5):
            for center, context in stream.iter_batches(batch_size=8192, epoch=epoch):
                ...
    """

    def __init__(
        self,
        tokens: np.ndarray,
        sent_lengths: np.ndarray,
        vocab: Vocab,
        window: int = 5,
        arch: str = "skipgram",
        subsample: float = 0.0,
        seed: int = 0,
        chunk_pairs: int = 4_000_000,
    ) -> None:
        assert arch in ("skipgram", "cbow"), f"未知结构: {arch}"
        self.tokens = np.asarray(tokens, dtype=np.int32)
        self.sent_lengths = np.asarray(sent_lengths, dtype=np.int64)
        self.vocab = vocab
        self.window = int(window)
        self.arch = arch
        self.subsample = float(subsample or 0.0)
        self.seed = int(seed)
        self.chunk_pairs = int(chunk_pairs)

        # 窗口偏移（去掉 0），形状 (2W,)
        offs = np.arange(-self.window, self.window + 1)
        self.offsets = offs[offs != 0].astype(np.int64)
        self.offsets_abs = np.abs(self.offsets)

        # 每个 token id 的保留概率
        self.keep_prob = self._build_keep_prob()

        # 每个 epoch 准备
        self._epoch: Optional[int] = None
        self.tokens_e: np.ndarray = self.tokens
        self.sent_starts: np.ndarray = np.zeros(0, dtype=np.int64)
        self.sent_ends: np.ndarray = np.zeros(0, dtype=np.int64)
        self._prepared_size = -1

    # ---------------- 高频词下采样 ----------------
    def _build_keep_prob(self) -> np.ndarray:
        counts = self.vocab.counts.astype(np.float64)
        total = counts.sum()
        keep = np.ones(len(self.vocab), dtype=np.float64)
        if self.subsample and self.subsample > 0 and total > 0:
            f = counts / total
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = self.subsample / f
                p = np.sqrt(ratio) + ratio
            keep = np.clip(p, 0.0, 1.0)
            keep[~np.isfinite(keep)] = 1.0
        keep[UNK_IDX] = 1.0
        keep[PAD_IDX] = 0.0
        return keep.astype(np.float64)

    # ---------------- epoch 准备 ----------------
    def _prepare_epoch(self, epoch: int, rng: np.random.Generator) -> None:
        if self._epoch == epoch:
            return
        self._epoch = epoch

        lengths = self.sent_lengths
        if self.subsample and self.subsample > 0:
            # 1) 按 keep_prob 抽样 token
            rnd = rng.random(len(self.tokens))
            mask = rnd < self.keep_prob[self.tokens]
            tokens_e = self.tokens[mask]
            # 2) 重建句子边界（先给每个 token 打上句子号，再 bincount）
            sent_id = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)[mask]
            new_lengths = np.bincount(sent_id, minlength=len(lengths))
        else:
            tokens_e = self.tokens
            new_lengths = lengths

        new_lengths = new_lengths[new_lengths > 0]
        if len(tokens_e) == 0 or len(new_lengths) == 0:
            raise RuntimeError("下采样后语料为空，请调大 data.subsample 或关闭它")

        ends = np.cumsum(new_lengths)
        starts = np.concatenate([[0], ends[:-1]])
        self.tokens_e = tokens_e
        self.sent_starts = starts
        self.sent_ends = ends
        self._prepared_size = len(tokens_e)

    def prepare(self, epoch: int = 0, seed: Optional[int] = None) -> None:
        """提前准备某个 epoch（下采样 + 重算句子边界），使统计量立即可用。"""
        rng = np.random.default_rng((seed if seed is not None else self.seed)
                                    + epoch * 1_000_003)
        self._prepare_epoch(epoch, rng)

    # ---------------- 统计 ----------------
    @property
    def tokens_per_epoch(self) -> int:
        return len(self.tokens_e) if self._prepared_size > 0 else len(self.tokens)

    @property
    def pairs_per_epoch(self) -> int:
        return int(self.tokens_per_epoch * (self.window + 1))

    # ---------------- 核心：生成一个 chunk 的样本 ----------------
    def _make_chunk(self, centers: np.ndarray, rng: np.random.Generator):
        """给定中心词位置，生成对应的样本对。"""
        tokens_e = self.tokens_e
        n = len(tokens_e)

        # 找每个中心词所属的句子，并取其边界
        sent = np.searchsorted(self.sent_ends, centers, side="right")
        s_start = self.sent_starts[sent]
        s_end = self.sent_ends[sent]

        # 动态窗口：r ~ U{1..W}
        r = rng.integers(1, self.window + 1, size=len(centers))

        pos = centers[:, None] + self.offsets[None, :]            # (M, 2W)
        valid = (self.offsets_abs[None, :] <= r[:, None]) & \
                (pos >= s_start[:, None]) & (pos < s_end[:, None])

        # centers 是「位置」，必须先映射回 token id 才能当标签用
        center_ids = tokens_e[centers].astype(np.int64)

        if self.arch == "skipgram":
            per_center = valid.sum(axis=1)
            centers_rep = np.repeat(center_ids, per_center)
            ctx = tokens_e[np.clip(pos, 0, n - 1)][valid].astype(np.int64)
        else:  # cbow
            rows, cols = np.nonzero(valid)
            rank = (np.cumsum(valid, axis=1) - 1)[rows, cols]      # 行内序号
            ctx = np.full((len(centers), len(self.offsets)), PAD_IDX, dtype=np.int64)
            ctx[rows, rank] = tokens_e[np.clip(pos[rows, cols], 0, n - 1)]
            centers_rep = center_ids

        # chunk 内再打乱一次，避免窗口顺序带来的相关性
        perm = rng.permutation(len(centers_rep))
        return centers_rep[perm], ctx[perm]

    # ---------------- 对外接口 ----------------
    def iter_batches(self, batch_size: int, epoch: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """按 batch 产出 (center, context)。

        一个 epoch = 把语料里的每个 token 恰好当一次中心词（位置随机打乱）。
        """
        rng = np.random.default_rng(self.seed + epoch * 1_000_003)
        self._prepare_epoch(epoch, rng)

        n = len(self.tokens_e)
        order = rng.permutation(n)          # 一个 epoch = 一次完整遍历
        chunk_centers = max(batch_size * 8, self.chunk_pairs // (self.window + 1))

        buf_c = np.zeros(0, dtype=np.int64)
        buf_x = np.zeros(0, dtype=np.int64) if self.arch == "skipgram" \
            else np.zeros((0, len(self.offsets)), dtype=np.int64)

        cursor = 0
        while cursor < n:
            take = min(chunk_centers, n - cursor)
            centers = np.sort(order[cursor:cursor + take])   # 排序后取值更贴近内存顺序
            cursor += take
            c, x = self._make_chunk(centers, rng)

            buf_c = np.concatenate([buf_c, c]) if buf_c.size else c
            buf_x = np.concatenate([buf_x, x]) if len(buf_x) else x

            while len(buf_c) >= batch_size:
                yield buf_c[:batch_size], buf_x[:batch_size]
                buf_c = buf_c[batch_size:]
                buf_x = buf_x[batch_size:]

        if len(buf_c) > 0:
            yield buf_c, buf_x


# --------------------------------------------------------------------------
# 自检 / 教学用：打印几组样本长什么样
# --------------------------------------------------------------------------
def preview(stream: PairStream, k: int = 12, batch_size: int = 512) -> None:
    itos = stream.vocab.itos
    print(f"\n结构: {stream.arch}  窗口: ±{stream.window}  "
          f"subsample: {stream.subsample}")
    print(f"每 epoch 有效 token: {human_int(stream.tokens_per_epoch)}  "
          f"预计样本对: {human_int(stream.pairs_per_epoch)}")
    print("\n样本示例（中心词 -> 上下文）:")
    shown = 0
    for center, context in stream.iter_batches(batch_size=batch_size, epoch=0):
        for i in range(len(center)):
            if stream.arch == "skipgram":
                print(f"  {itos[int(center[i])]:<16s} -> {itos[int(context[i])]}")
            else:
                ctx_words = [itos[int(t)] for t in context[i] if int(t) != PAD_IDX]
                print(f"  {' '.join(ctx_words):<34s} -> {itos[int(center[i])]}")
            shown += 1
            if shown >= k:
                return


def main() -> None:
    from .preprocess import load_processed
    from .utils import force_utf8_stdout, load_config

    force_utf8_stdout()
    p = argparse.ArgumentParser(description="训练样本生成（查看 / 自检）")
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--source", default=None, help="覆盖配置里的 data.source")
    p.add_argument("--lang", default=None)
    p.add_argument("--min-count", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--arch", default=None, choices=["skipgram", "cbow"])
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--show", type=int, default=12)
    a = p.parse_args()

    cfg = load_config(a.config)
    d = cfg["data"]
    source = a.source or d["source"]
    lang = a.lang or d.get("lang", "en")
    min_count = a.min_count if a.min_count is not None else d.get("min_count", 5)
    max_tokens = a.max_tokens if a.max_tokens is not None else d.get("max_tokens")

    tokens, sent_lengths, vocab, _ = load_processed(source, lang, min_count, max_tokens)
    m = cfg["model"]
    stream = PairStream(
        tokens, sent_lengths, vocab,
        window=a.window or m["window"],
        arch=a.arch or m["arch"],
        subsample=a.subsample if a.subsample is not None else m.get("subsample", 0.0),
        seed=cfg.get("seed", 42),
    )
    preview(stream, k=a.show)


if __name__ == "__main__":
    main()
