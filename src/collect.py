"""第 1 步：数据收集。

支持四种数据源：
    demo      —— 内置小语料（data/samples/demo_corpus.txt），离线可用，秒级
    text8     —— 经典 Word2Vec 基准语料，约 1700 万词（推荐正式训练）
    wikitext2 —— WikiText-2，约 200 万词（更小，想更快出结果可以用）
    local     —— 使用你自己的语料文件

统一产出： data/raw/<name>/raw.txt   （纯文本，一行一句或整篇皆可）
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from .utils import Timer, ensure_dir, format_hms, human_int, resolve_path

# --------------------------------------------------------------------------
# 数据源登记表：每个源给出候选 URL（按顺序重试，便于走不同镜像）
# --------------------------------------------------------------------------
SOURCES: Dict[str, Dict] = {
    "demo": {
        "kind": "local",
        "path": "data/samples/demo_corpus.txt",
        "desc": "内置小语料（约 1.5 万词），用于跑通流程与教学演示",
    },
    "text8": {
        "kind": "zip",
        "urls": [
            "http://mattmahoney.net/dc/text8.zip",
            "https://data.deepai.org/text8.zip",
        ],
        "inner": "text8",
        "desc": "text8：Wikipedia 清洗后的 1e8 字符 / 约 1700 万词，Word2Vec 经典基准",
    },
    "wikitext2": {
        "kind": "zip",
        "urls": [
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
            "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1.zip",
        ],
        "inner": None,  # 解压后合并目录下所有 .txt
        "concat": ["train.txt", "valid.txt", "test.txt"],
        "desc": "WikiText-2 raw：约 200 万词，规模小、质量高",
    },
}

UNZIP_ARCHIVE = "download.zip"


def _progress_hook(name: str):
    state = {"last": 0}

    def hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        if pct - state["last"] >= 1.0:
            state["last"] = pct
            sys.stdout.write(
                f"\r  下载 {name}: {pct:5.1f}%  "
                f"({human_int(downloaded)}B / {human_int(total_size)}B)"
            )
            sys.stdout.flush()

    return hook


def remote_size(url: str, timeout: float = 20.0) -> Optional[int]:
    """用 HEAD 请求问出文件大小；服务器不支持 HEAD 时返回 None。"""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001
        return None


def _expected_size(urls: List[str]) -> Optional[int]:
    for url in urls:
        size = remote_size(url)
        if size:
            return size
    return None


def download(urls: List[str], dest: Path, retries_per_url: int = 2) -> Path:
    """依次尝试候选 URL 下载；全部失败则给出明确的手动下载指引。

    会校验文件大小：下载被中断（Ctrl+C、断网、SSH 掉线）留下的残缺文件
    不会被当成已完成 —— 否则后面会拿着半个语料训练，还很难发现。
    """
    expected = _expected_size(urls)

    if dest.exists() and dest.stat().st_size > 0:
        local = dest.stat().st_size
        if expected is None or local >= expected:
            print(f"  已存在且完整，跳过下载: {dest}")
            return dest
        print(f"  发现不完整的文件（{human_int(local)}B / {human_int(expected)}B），重新下载")
        dest.unlink()
    ensure_dir(dest.parent)

    for url in urls:
        for attempt in range(1, retries_per_url + 1):
            try:
                print(f"  尝试下载 ({attempt}/{retries_per_url}): {url}")
                urllib.request.urlretrieve(url, dest, reporthook=_progress_hook(dest.name))
                print()
                got = dest.stat().st_size
                if got == 0:
                    raise IOError("下载文件为空")
                if expected and got < expected:
                    raise IOError(
                        f"下载不完整: {human_int(got)}B < {human_int(expected)}B"
                    )
                return dest
            except Exception as exc:  # noqa: BLE001
                print(f"  ！{exc}")
                if dest.exists():
                    dest.unlink()
    raise RuntimeError(
        "所有下载源都失败了。\n"
        "网络受限时请手动下载后放到 data/raw/ 下，再重跑本脚本：\n"
        f"  目标文件名: {dest}\n"
        "  text8 下载地址示例: http://mattmahoney.net/dc/text8.zip"
    )


def _extract_zip(zip_path: Path, out_dir: Path) -> List[Path]:
    print(f"  解压: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
        return [out_dir / n for n in zf.namelist()]


def collect(source: str, out_root: str = "data/raw", local_path: Optional[str] = None,
            force: bool = False) -> Path:
    """收集数据，返回 raw.txt 的路径。"""
    if source not in SOURCES and source != "local":
        raise ValueError(f"未知数据源: {source}，可选: {list(SOURCES) + ['local']}")

    out_dir = ensure_dir(resolve_path(f"{out_root}/{source}"))
    raw_txt = out_dir / "raw.txt"
    if raw_txt.exists() and not force:
        print(f"已存在: {raw_txt}（用 --force 重新收集）")
        return raw_txt

    timer = Timer()
    print(f"== 数据收集: {source} ==")

    if source == "local":
        src = resolve_path(local_path)
        if src is None or not src.exists():
            raise FileNotFoundError(f"本地语料不存在: {local_path}")
        shutil.copyfile(src, raw_txt)
    else:
        spec = SOURCES[source]
        if spec["kind"] == "local":
            if source == "demo":
                # demo 语料由生成器扩充，保证离线也能有足够训练量
                from .demo_corpus import write_demo_corpus

                n = write_demo_corpus(raw_txt)
                print(f"  已生成离线演示语料，约 {n} 个词")
            else:
                src = resolve_path(spec["path"])
                shutil.copyfile(src, raw_txt)
        else:
            archive = out_dir / UNZIP_ARCHIVE
            download(spec["urls"], archive)
            members = _extract_zip(archive, out_dir)

            concat: Optional[List[str]] = spec.get("concat")
            if concat:
                parts = [out_dir / name for name in concat if (out_dir / name).exists()]
                if not parts:
                    raise RuntimeError(f"解压后未找到预期文件: {concat}")
            else:
                inner = spec.get("inner")
                target = out_dir / inner if inner else None
                if target is not None and target.exists():
                    parts = [target]
                else:
                    parts = [p for p in members if p.suffix in (".txt", "") and p.is_file()]

            print(f"  合并 {len(parts)} 个文件 -> raw.txt")
            with open(raw_txt, "w", encoding="utf-8") as fout:
                for p in parts:
                    with open(p, "r", encoding="utf-8", errors="ignore") as fin:
                        shutil.copyfileobj(fin, fout)
                    fout.write("\n")
            for p in members:
                try:
                    if p.is_file() and p != raw_txt:
                        p.unlink()
                except OSError:
                    pass
            if archive.exists():
                archive.unlink()

    size_mb = raw_txt.stat().st_size / 1024 ** 2
    print(f"完成: {raw_txt}  ({size_mb:.1f} MB)  用时 {format_hms(timer.elapsed)}")
    print(f"下一步: python -m src.preprocess --source {source}")
    return raw_txt


def main() -> None:
    from .utils import force_utf8_stdout

    force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Word2Vec 数据收集")
    parser.add_argument("--source", default="demo",
                        choices=list(SOURCES) + ["local"], help="数据源")
    parser.add_argument("--out", default="data/raw", help="原始数据输出目录")
    parser.add_argument("--local-path", default=None, help="source=local 时的语料路径")
    parser.add_argument("--force", action="store_true", help="强制重新收集")
    parser.add_argument("--list", action="store_true", help="列出可用数据源")
    args = parser.parse_args()

    if args.list:
        for name, spec in SOURCES.items():
            print(f"  {name:10s} {spec['desc']}")
        print(f"  {'local':10s} 使用你自己的语料文件（--local-path 指定）")
        return

    collect(args.source, args.out, args.local_path, args.force)


if __name__ == "__main__":
    main()
