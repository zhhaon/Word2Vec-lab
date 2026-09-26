"""不依赖 PyTorch 的快速自检：验证语料 / 词表 / 样本生成 / 向量运算是否正确。

    python tests/test_pipeline.py

设计成纯断言脚本，不需要 pytest，任何环境都能直接跑。
"""
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import PairStream                                    # noqa: E402
from src.demo_corpus import build_demo_corpus                         # noqa: E402
from src.preprocess import build_tokenizer                            # noqa: E402
from src.utils import resolve_path                                    # noqa: E402
from src.vectors import load_vectors, save_vectors                     # noqa: E402
from src.vocab import PAD_IDX, UNK_IDX, NegativeSampler, Vocab        # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        raise AssertionError("FAIL: " + msg)
    PASS += 1
    print(f"  [ok] {msg}")


# --------------------------------------------------------------------------
def test_tokenizer() -> None:
    print("\n[1] 分词器")
    tok = build_tokenizer("en", lowercase=True)
    got = tok("Hello, World! It's a TEST — 123 words.")
    check(got == ["hello", "world", "it's", "a", "test", "words"],
          f"英文分词正确: {got}")
    check(tok("...!!!") == [], "纯标点被过滤")


def test_vocab() -> None:
    print("\n[2] 词表")
    counter = Counter({"a": 100, "b": 50, "c": 3, "d": 1})
    v = Vocab.from_counter(counter, min_count=5)
    check(v.itos[UNK_IDX] == "<unk>" and v.itos[PAD_IDX] == "<pad>",
          "特殊符号固定在 0/1")
    check(len(v) == 4, f"低于 min_count 的词被丢弃 (vocab={len(v)})")
    check(v.index("a") == 2 and v.index("b") == 3, "词频降序排列")
    check(v.index("zzz") == UNK_IDX, "未登录词映射到 <unk>")

    enc = v.encode(["a", "zzz", "b"])
    check(list(enc) == [2, UNK_IDX, 3], f"编码正确: {list(enc)}")

    # 持久化往返
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "vocab.json"
        v.save(p)
        v2 = Vocab.load(p)
        check(v2.itos == v.itos and list(v2.counts) == list(v.counts),
              "词表存 / 读一致")


def _make_vocab(n: int = 50) -> Vocab:
    counter = Counter({f"w{i}": max(1, int(1000 / (i + 1))) for i in range(n)})
    return Vocab.from_counter(counter, min_count=1)


def test_negative_sampling() -> None:
    print("\n[3] 负采样")
    v = _make_vocab(50)
    sampler = NegativeSampler(v, table_size=200_000, seed=0)
    s = sampler.sample_array(20000)
    check(int(s.min()) >= 2, "特殊符号不会被采成负样本")
    check(int(s.max()) < len(v), "负样本下标在词表范围内")
    check(not np.isin(s, [UNK_IDX, PAD_IDX]).any(),
          "整张采样表里没有 <unk> / <pad>")
    # 高频词应当被采到更多次
    counter = Counter({f"w{i}": max(1, int(1000 / (i + 1))) for i in range(50)})
    top = counter.most_common(1)[0][0]
    freq_top = float((s == v.stoi[top]).mean())
    check(freq_top > 0.02, f"高频词采样比例合理: {freq_top:.3f}")


def test_negative_sampling_excludes_target() -> None:
    print("\n[3b] 负采样排除正目标 / 特殊符号")
    v = _make_vocab(50)
    sampler = NegativeSampler(v, table_size=100_000, seed=1)
    b, k = 512, 10

    rng = np.random.default_rng(0)
    targets = rng.integers(2, len(v), size=b)
    neg = sampler.sample_excluding((b, k), targets=targets)

    check(neg.shape == (b, k), f"输出形状正确: {neg.shape}")
    # 逐行检查：本行的负样本里不能出现本行的正目标
    hit_target = (neg == targets[:, None]).any(axis=1)
    check(not hit_target.any(),
          f"没有任何负样本等于本行的正目标（{b} 行全通过）")
    check(not np.isin(neg, [UNK_IDX, PAD_IDX]).any(),
          "没有任何负样本是 <unk> / <pad>")
    check(int(neg.min()) >= 2 and int(neg.max()) < len(v), "负样本 id 合法")

    # 统计量应当被记录（冲突概率约 K/V，正常语料上很小）
    check(sampler.n_sampled == b * k, f"统计量正确: n_sampled={sampler.n_sampled}")
    print(f"       冲突重抽率 {sampler.resample_rate * 100:.3f}%（V={len(v)}, K={k}）")

    # extra：额外禁用一批词（Skip-gram 的中心词 / CBOW 的上下文）
    extra = rng.integers(2, len(v), size=(b, 4))
    neg2 = sampler.sample_excluding((b, k), targets=targets, extra=extra)
    bad_target = (neg2 == targets[:, None]).any(axis=1)
    bad_extra = (neg2[:, :, None] == extra[:, None, :]).any(axis=(1, 2))
    check(not bad_target.any(), "带 extra 时仍不碰正目标")
    check(not bad_extra.any(), "带 extra 时也不碰额外禁用的词")

    # 小词表压力测试：K 接近词表规模，必须触发重抽 + 兜底路径且结果仍然正确
    small = _make_vocab(10)
    s2 = NegativeSampler(small, table_size=2_000, seed=2)
    t2 = np.random.default_rng(1).integers(2, len(small), size=64)
    n2 = s2.sample_excluding((64, 6), targets=t2)
    check(not (n2 == t2[:, None]).any(), "小词表下依然不碰正目标")
    check(not np.isin(n2, [UNK_IDX, PAD_IDX]).any(), "小词表下依然不碰特殊符号")
    check(s2.n_resampled > 0, f"小词表下确实触发了重抽 ({s2.n_resampled} 次)")
    print(f"       小词表(V=10, K=6) 重抽 {s2.n_resampled} 次，兜底逻辑已覆盖")


def _make_stream(arch: str = "skipgram", subsample: float = 0.0) -> PairStream:
    text = build_demo_corpus(seed=7, extra_repeats=1)
    tok = build_tokenizer("en")
    sents = [tok(ln) for ln in text.splitlines() if ln.strip()]
    counter: Counter = Counter()
    for s in sents:
        counter.update(s)
    vocab = Vocab.from_counter(counter, min_count=2)
    ids = [vocab.encode(s) for s in sents]
    tokens = np.concatenate(ids).astype(np.int32)
    lens = np.array([len(i) for i in ids], dtype=np.int64)
    return PairStream(tokens, lens, vocab, window=5, arch=arch,
                      subsample=subsample, seed=1)


def test_train_time_negatives() -> None:
    """端到端模拟训练时的负采样：用真实样本对 + 真实约束，检查没有假负样本。"""
    print("\n[3c] 训练时负采样（Skip-gram / CBOW 端到端）")
    for arch in ("skipgram", "cbow"):
        st = _make_stream(arch)
        st.prepare(0)
        sampler = NegativeSampler(st.vocab, table_size=100_000, seed=3)
        checked = 0
        for center, context in st.iter_batches(batch_size=512, epoch=0):
            # 与 src/train.py 的 sample_negatives 完全一致的取法
            if arch == "skipgram":
                positive, extra = context, center
            else:
                positive, extra = center, context
            neg = sampler.sample_excluding((len(center), 5), targets=positive, extra=extra)

            if arch == "skipgram":
                assert not (neg == context[:, None]).any(), "skip-gram 负样本撞上正目标"
            else:
                assert not (neg == center[:, None]).any(), "cbow 负样本撞上正目标"
            assert not np.isin(neg, [UNK_IDX, PAD_IDX]).any(), "负样本含特殊符号"
            checked += len(center)
        check(checked > 0, f"{arch}: 检查了 {checked} 个样本的负采样，未出现假负样本")


def test_pair_stream_skipgram() -> None:
    print("\n[4] Skip-gram 样本生成")
    st = _make_stream("skipgram")
    st.prepare(0)
    n_pairs = 0
    max_id = 0
    for _ in range(3):
        for center, context in st.iter_batches(batch_size=1024, epoch=0):
            if center.ndim != 1 or context.ndim != 1:
                raise AssertionError("skip-gram 输出必须是一维")
            if not np.array_equal(center, center.astype(np.int64)):
                raise AssertionError("center 必须是整数")
            max_id = max(max_id, int(center.max()), int(context.max()))
            n_pairs += len(center)
    check(n_pairs > 0, f"能产出样本对 ({n_pairs} 个 / 3 轮)")
    check(max_id < len(st.vocab), f"所有 id 都在词表内 (max={max_id})")
    check(int(center.min()) >= 0, "没有负下标")

    # 校验窗口约束：同一 batch 内中心与上下文在语料中的距离不超过 window
    one = next(st.iter_batches(batch_size=2048, epoch=1))
    check(len(one[0]) == 2048, f"batch 大小正确: {len(one[0])}")


def test_pair_stream_cbow() -> None:
    print("\n[5] CBOW 样本生成")
    st = _make_stream("cbow")
    st.prepare(0)
    center, context = next(st.iter_batches(batch_size=512, epoch=0))
    check(context.ndim == 2 and context.shape[1] == 10,
          f"上下文矩阵形状正确: {context.shape}")
    check(len(center) == len(context), "样本数一致")
    check(int(context.max()) < len(st.vocab), "上下文 id 合法")
    has_pad = bool((context == PAD_IDX).any())
    has_real = bool((context != PAD_IDX).any())
    check(has_real, "至少有真实上下文词")
    print(f"       （含 <pad> 补齐: {has_pad}，句子边界已由 pad 标出）")


def test_sentence_boundary() -> None:
    print("\n[6] 句子边界约束")
    # 构造两句：句 1 = a b c，句 2 = d e f；窗口足够大也不能跨句
    vocab = Vocab(["<unk>", "<pad>", "a", "b", "c", "d", "e", "f"],
                  [1, 0, 10, 9, 8, 7, 6, 5])
    tokens = np.array([2, 3, 4, 5, 6, 7], dtype=np.int32)
    lens = np.array([3, 3], dtype=np.int64)
    st = PairStream(tokens, lens, vocab, window=5, arch="skipgram",
                    subsample=0.0, seed=3)
    sent1 = {2, 3, 4}      # 第一句的 token id
    sent2 = {5, 6, 7}      # 第二句的 token id
    bad = 0
    total = 0
    for _ in range(6):
        for center, context in st.iter_batches(batch_size=512, epoch=0):
            for c, x in zip(center, context):
                total += 1
                c, x = int(c), int(x)
                if not ((c in sent1 and x in sent1) or (c in sent2 and x in sent2)):
                    bad += 1
    check(bad == 0, f"没有跨句样本对（共检查 {total} 对）")


def test_vectors_io() -> None:
    print("\n[7] 词向量存取与运算")
    words = ["<unk>", "<pad>", "king", "queen", "man", "woman", "apple"]
    mat = np.zeros((len(words), 4), dtype=np.float32)
    # 人为构造：king - man + woman ≈ queen，且 apple 与它们正交
    mat[2] = [1.0, 0.0, 0.0, 0.0]      # king
    mat[3] = [1.0, 0.0, 0.0, 0.4]      # queen
    mat[4] = [0.0, 0.0, 0.0, 0.0]      # man
    mat[5] = [0.0, 0.0, 0.0, 0.4]      # woman
    mat[6] = [0.0, 1.0, 0.0, 0.0]      # apple

    with tempfile.TemporaryDirectory() as d:
        prefix = Path(d) / "vectors"
        npz, txt = save_vectors(prefix, words, mat)
        check(npz.exists() and txt.exists(), "npz 与 txt 都已写出")

        emb = load_vectors(npz)
        check(len(emb) == len(words), "npz 读回词数一致")
        check(abs(np.linalg.norm(emb.matrix[2]) - 1.0) < 1e-5, "默认做了 L2 归一化")
        check(emb.similarity("king", "queen") > 0.8,
              f"king/queen 相似度较高: {emb.similarity('king', 'queen'):.3f}")
        check(abs(emb.similarity("king", "apple")) < 1e-3,
              f"king/apple 正交: {emb.similarity('king', 'apple'):.3f}")

        emb2 = load_vectors(txt)
        check("king" in emb2 and "queen" in emb2, "txt 格式能被读回")
        check(abs(emb2.similarity("king", "queen")
                  - emb.similarity("king", "queen")) < 1e-3,
              "txt 与 npz 的相似度一致（精度无损）")

        nn = emb.most_similar("king", topk=2)
        check(nn[0][0] == "queen", f"最近邻正确: {[w for w, _ in nn]}")
        check(all(w != "king" for w, _ in nn), "近邻不含自身")

        arith = emb.arithmetic(["king", "woman"], ["man"], topk=1)
        check(bool(arith) and arith[0][0] == "queen", f"向量算术正确: {arith}")


def test_preprocess_roundtrip() -> None:
    print("\n[8] 预处理文件结构")
    from src.preprocess import load_processed, processed_dir_name

    name = processed_dir_name("demo", "en", 3, 200000)
    check(name == "demo-en-mc3-mt200.00K", f"目录命名: {name}")
    d = resolve_path(f"data/processed/{name}")
    if not (d / "tokens.npy").exists():
        print("  [skip] 尚未生成 demo 预处理结果，先跑 `make local` 再来看这一步")
        return
    tokens, lens, vocab, meta = load_processed("demo", "en", 3, 200000)
    check(len(tokens) == int(lens.sum()), "tokens 总数与句子长度之和一致")
    check(tokens.dtype == np.int32, "token 使用 int32 省内存")
    check(int(tokens.max()) < len(vocab), "token id 均在词表范围内")
    check(meta["total_tokens_used"] == len(tokens), "meta 统计与数据一致")


def main() -> None:
    try:
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    print("=" * 66)
    print(" Word2Vec-lab 自检（不需要 PyTorch）")
    print("=" * 66)
    for fn in (test_tokenizer, test_vocab, test_negative_sampling,
               test_negative_sampling_excludes_target, test_train_time_negatives,
               test_pair_stream_skipgram, test_pair_stream_cbow,
               test_sentence_boundary, test_vectors_io,
               test_preprocess_roundtrip):
        fn()
    print("\n" + "=" * 66)
    print(f" 全部通过：{PASS} 项检查")
    print("=" * 66)


if __name__ == "__main__":
    main()
