"""第 2 步：数据清理与预处理。

流程：
    raw.txt  ->  逐句分词  ->  （第一次遍历）词频统计  ->  构建词表
             ->  （第二次遍历）编码成 int32 ID 序列  ->  落盘

产出（data/processed/<name>/）：
    tokens.npy        int32，全语料的 token id 序列（已按 max_tokens 截断）
    sent_lengths.npy  int32，每句的 token 数（用于按句打乱 & 窗口不越句）
    vocab.json        词表（itos + counts）
    meta.json         统计信息（词表大小、覆盖率等）

设计要点：
    * 全部用流式分块处理，1700 万词级别语料内存占用 < 1GB；
    * 保留句子边界，训练时窗口不会跨越句号乱连；
    * <unk> / <pad> 固定占据下标 0 / 1。
"""
from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np

from .utils import ensure_dir, human_int, resolve_path, save_json
from .vocab import UNK_IDX, Vocab

# 英文 token 规则：小写字母串，允许内部单引号（如 don't）
EN_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
# 中文/混合文本：先粗切出「中文串 + 英数串」，再交给 jieba
ZH_KEEP_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")
# 控制字符 / 零宽字符
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\ufeff]")

PSEUDO_SENT_TOKENS = 1000   # 无换行的长文本，按此长度切成伪句子
CHUNK_BYTES = 8 << 20       # 8MB


# --------------------------------------------------------------------------
# 分词器
# --------------------------------------------------------------------------
def build_tokenizer(lang: str = "en", lowercase: bool = True) -> Callable[[str], List[str]]:
    lang = lang.lower()
    if lang == "en":
        def tokenize_en(text: str) -> List[str]:
            if lowercase:
                text = text.lower()
            return EN_TOKEN_RE.findall(text)

        return tokenize_en

    if lang == "zh":
        import jieba  # 延迟导入：只有处理中文时才需要

        def tokenize_zh(text: str) -> List[str]:
            out: List[str] = []
            for seg in ZH_KEEP_RE.findall(text):
                if seg.isascii():
                    out.append(seg.lower() if lowercase else seg)
                else:
                    out.extend(jieba.lcut(seg))
            return [t for t in out if t.strip()]

        return tokenize_zh

    raise ValueError(f"不支持的 lang: {lang}，可选 en / zh")


# --------------------------------------------------------------------------
# 流式读取 + 逐句分词
# --------------------------------------------------------------------------
def _split_raw_by_whitespace(text: str, max_bytes: int) -> Iterator[str]:
    """把超长的一行按空白边界切成安全长度的片段，避免切碎单词。"""
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_bytes, n)
        if end < n:
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end),
                      text.rfind("\t", start, end))
            if cut > start:
                end = cut + 1
        yield text[start:end]
        start = end


def _emit_pseudo(tokens: List[str], size: int) -> Iterator[List[str]]:
    if not tokens:
        return
    if len(tokens) <= size:
        yield tokens
        return
    for i in range(0, len(tokens), size):
        yield tokens[i:i + size]


def iter_sentence_token_lists(
    path: Path,
    tokenize: Callable[[str], List[str]],
    pseudo_sent_tokens: int = PSEUDO_SENT_TOKENS,
    chunk_bytes: int = CHUNK_BYTES,
) -> Iterator[List[str]]:
    """逐句产出 token 列表。真实换行 = 句子边界；超长行按伪句子切分。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = CTRL_RE.sub("", line)
            if len(line) <= chunk_bytes:
                yield from _emit_pseudo(tokenize(line), pseudo_sent_tokens)
            else:
                # 超长行（例如 text8 整篇就是一行）：
                # 按空白边界切成若干片段再分词，然后用「游标 + 每片回收一次」的方式
                # 划出伪句子。注意不能用 pending = pending[size:] 反复切片 ——
                # 那是 O(n) 拷贝，片段里上百万个 token 会把这一步拖成几十秒。
                pending: List[str] = []
                start = 0
                for piece in _split_raw_by_whitespace(line, chunk_bytes):
                    pending.extend(tokenize(piece))
                    while len(pending) - start >= pseudo_sent_tokens:
                        yield pending[start:start + pseudo_sent_tokens]
                        start += pseudo_sent_tokens
                    if start:
                        pending = pending[start:]
                        start = 0
                if start:
                    pending = pending[start:]
                yield from _emit_pseudo(pending, pseudo_sent_tokens)


# --------------------------------------------------------------------------
# 目录命名
# --------------------------------------------------------------------------
def processed_dir_name(source: str, lang: str, min_count: int,
                       max_tokens: Optional[int]) -> str:
    mt = "all" if not max_tokens else human_int(max_tokens)
    return f"{source}-{lang}-mc{min_count}-mt{mt}"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def preprocess(
    source: str = "demo",
    lang: str = "en",
    raw_dir: str = "data/raw",
    out_root: str = "data/processed",
    min_count: int = 5,
    max_vocab: Optional[int] = None,
    max_tokens: Optional[int] = None,
    force: bool = False,
) -> Path:
    raw_txt = resolve_path(f"{raw_dir}/{source}/raw.txt")
    if not raw_txt.exists():
        raise FileNotFoundError(
            f"找不到原始语料 {raw_txt}，请先运行: python -m src.collect --source {source}"
        )

    out_dir = ensure_dir(resolve_path(
        f"{out_root}/{processed_dir_name(source, lang, min_count, max_tokens)}"
    ))
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        print(f"已存在: {out_dir}（用 --force 重新处理）")
        return out_dir

    t0 = time.time()
    tokenize = build_tokenizer(lang, lowercase=True)

    # ---------- 第一次遍历：词频统计 ----------
    print("== 第 2 步：数据清理与预处理 ==")
    print(f"[1/2] 统计词频 ...  ({raw_txt})")
    counter: Counter = Counter()
    total_raw = 0
    n_sent = 0
    for toks in iter_sentence_token_lists(raw_txt, tokenize):
        counter.update(toks)
        total_raw += len(toks)
        n_sent += 1
    print(f"      总词数 {human_int(total_raw)}，句子数 {n_sent}，"
          f"去重后 {human_int(len(counter))} 个词形")

    if total_raw == 0:
        raise ValueError("分词后没有任何 token，请检查语料与 --lang 设置")

    vocab = Vocab.from_counter(counter, min_count=min_count, max_vocab=max_vocab)
    print(f"      词表大小 {human_int(len(vocab))}（min_count={min_count}）")

    # ---------- 第二次遍历：编码 ----------
    print("[2/2] 编码为 ID 序列 ...")
    stoi = vocab.stoi
    unk = UNK_IDX
    id_blocks: List[np.ndarray] = []
    len_blocks: List[np.ndarray] = []
    used = 0
    truncated = False

    for toks in iter_sentence_token_lists(raw_txt, tokenize):
        if not toks:
            continue
        ids = np.fromiter((stoi.get(w, unk) for w in toks), dtype=np.int32, count=len(toks))
        if max_tokens is not None and used + len(ids) > max_tokens:
            ids = ids[: max_tokens - used]
            truncated = True
        if len(ids) == 0:
            break
        id_blocks.append(ids)
        len_blocks.append(np.array([len(ids)], dtype=np.int32))
        used += len(ids)
        if truncated:
            break

    tokens = np.concatenate(id_blocks) if id_blocks else np.zeros(0, dtype=np.int32)
    sent_lengths = np.concatenate(len_blocks) if len_blocks else np.zeros(0, dtype=np.int32)

    # ---------- 落盘 ----------
    np.save(out_dir / "tokens.npy", tokens)
    np.save(out_dir / "sent_lengths.npy", sent_lengths)
    vocab.save(out_dir / "vocab.json")

    unk_cnt = int((tokens == unk).sum())
    meta: Dict = {
        "source": source,
        "lang": lang,
        "raw_file": str(raw_txt),
        "total_tokens_raw": int(total_raw),
        "total_tokens_used": int(len(tokens)),
        "num_sentences": int(len(sent_lengths)),
        "avg_sentence_len": float(sent_lengths.mean()) if len(sent_lengths) else 0.0,
        "vocab_size": len(vocab),
        "min_count": min_count,
        "max_vocab": max_vocab,
        "max_tokens": max_tokens,
        "truncated": bool(truncated),
        "unk_rate": unk_cnt / max(1, len(tokens)),
        "unk_count": unk_cnt,
        "tokenizer": f"{lang}",
    }
    save_json(meta, meta_path)

    print(f"      写入 {out_dir}")
    print(f"        tokens.npy       {len(tokens):,} 个 token "
          f"({tokens.nbytes / 1024 ** 2:.1f} MB)")
    print(f"        sent_lengths.npy {len(sent_lengths):,} 句")
    print(f"        vocab.json       {len(vocab):,} 词")
    print(f"      <unk> 占比 {meta['unk_rate'] * 100:.2f}%（越低说明 min_count 越合适）")
    print("      高频词 Top10: " + ", ".join(vocab.itos[2:12]))
    print(f"完成，用时 {time.time() - t0:.1f}s")
    print("下一步: python -m src.train --config <你的配置>.yaml")
    return out_dir


def load_processed(source: str, lang: str = "en", min_count: int = 5,
                   max_tokens: Optional[int] = None, out_root: str = "data/processed"):
    """读取预处理结果，返回 (tokens, sent_lengths, vocab, meta)。"""
    d = resolve_path(f"{out_root}/{processed_dir_name(source, lang, min_count, max_tokens)}")
    if not (d / "tokens.npy").exists():
        raise FileNotFoundError(
            f"找不到预处理结果 {d}，请先运行: python -m src.preprocess --source {source} "
            f"--min-count {min_count}" + (f" --max-tokens {max_tokens}" if max_tokens else "")
        )
    import json

    tokens = np.load(d / "tokens.npy")
    sent_lengths = np.load(d / "sent_lengths.npy")
    vocab = Vocab.load(d / "vocab.json")
    with open(d / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return tokens, sent_lengths, vocab, meta


def main() -> None:
    from .utils import force_utf8_stdout

    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Word2Vec 数据清理与预处理")
    p.add_argument("--config", default=None,
                   help="从配置文件读取 data 段（推荐，保证与训练一致）")
    p.add_argument("--source", default=None, help="与 collect 的 --source 一致")
    p.add_argument("--lang", default=None, choices=["en", "zh"])
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--min-count", type=int, default=None, help="低于该词频的词并入 <unk>")
    p.add_argument("--max-vocab", type=int, default=None, help="词表上限")
    p.add_argument("--max-tokens", type=int, default=None, help="只保留前 N 个 token")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    cfg, d = {}, {}
    if a.config:
        from .utils import load_config

        cfg = load_config(a.config)
        d = cfg.get("data", {})

    source = a.source or d.get("source") or "demo"
    lang = a.lang or d.get("lang", "en")
    raw_dir = a.raw_dir or d.get("raw_dir", "data/raw")
    out = a.out or d.get("processed_dir", "data/processed")
    min_count = a.min_count if a.min_count is not None else int(d.get("min_count", 5))
    max_vocab = a.max_vocab if a.max_vocab is not None else d.get("max_vocab")
    max_tokens = a.max_tokens if a.max_tokens is not None else d.get("max_tokens")

    preprocess(source, lang, raw_dir, out, min_count, max_vocab, max_tokens, a.force)


if __name__ == "__main__":
    main()
