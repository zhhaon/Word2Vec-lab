"""公开评测数据集的下载、缓存与解析。

支持两种数据来源：

    1) 仓库内置的小评测集（data/eval/*.txt）
       —— 离线可用、秒级完成，用于跑通流程和训练途中快速评估；
    2) 公开标准数据集（按需下载并缓存到 data/eval/downloaded/）
       —— WordSim-353 / Google Analogy 等，正式训练结束时用它们出可对比的指标。

数据集登记在下面的 REGISTRY 里，加新数据集只要往里加一条。

⚠️ 关于 WordSim-353 的一个事实（容易让人以为解析写错了）：
    EN-WS-353-ALL.txt 有 353 行，但 (money, cash) 出现了两次（9.15 与 9.08），
    所以唯一词对是 352 个。本模块保留全部行、另外报告唯一词对数，不做静默去重。
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import ensure_dir, resolve_path

# --------------------------------------------------------------------------
# 下载源
# --------------------------------------------------------------------------
_WS353_BASE = ("https://raw.githubusercontent.com/mfaruqui/eval-word-vectors"
               "/master/data/word-sim/")

REGISTRY: Dict[str, Dict] = {
    "wordsim353": {
        "kind": "similarity",
        "filename": "EN-WS-353-ALL.txt",
        "urls": [_WS353_BASE + "EN-WS-353-ALL.txt"],
        # 主源挂掉时可以从相似性子集 + 相关性自己拼回全集：
        # 已验证 SIM ∪ REL 恰好等于 ALL（唯一词对 352 个）
        "rebuild_from": ["wordsim353-sim", "wordsim353-rel"],
        "desc": "WordSim-353 全量（353 行 / 352 个唯一词对，Finkelstein et al. 2001）",
    },
    "wordsim353-sim": {
        "kind": "similarity",
        "filename": "EN-WS-353-SIM.txt",
        "urls": [_WS353_BASE + "EN-WS-353-SIM.txt"],
        "desc": "WordSim-353 相似性子集（203 对）",
    },
    "wordsim353-rel": {
        "kind": "similarity",
        "filename": "EN-WS-353-REL.txt",
        "urls": [_WS353_BASE + "EN-WS-353-REL.txt"],
        "desc": "WordSim-353 相关性自集（252 对）",
    },
    "men": {
        "kind": "similarity",
        "filename": "EN-MEN-TR-3k.txt",
        "urls": [_WS353_BASE + "EN-MEN-TR-3k.txt"],
        "desc": "MEN 3000 对（Bruni et al. 2012）",
    },
    "simlex999": {
        "kind": "similarity",
        "filename": "EN-SIMLEX-999.txt",
        "urls": [_WS353_BASE + "EN-SIMLEX-999.txt"],
        "desc": "SimLex-999（Hill et al. 2015）",
    },
    "google-analogy": {
        "kind": "analogy",
        "filename": "questions-words.txt",
        "urls": [
            "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt",
            "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt",
        ],
        "desc": "Google Analogy Dataset（19544 题 / 14 类，Mikolov et al. 2013）",
    },
}

# 论文里的语义 / 句法分组，用于汇总
SEMANTIC_GROUPS = {
    "capital-common-countries", "capital-world", "currency",
    "city-in-state", "family",
}

# 内置小评测集（离线兜底）
BUNDLED_SIMILARITY = "data/eval/similarity_pairs.txt"
BUNDLED_ANALOGY = "data/eval/analogy_questions.txt"


def is_registered(name: str) -> bool:
    return name in REGISTRY


def cache_dir() -> Path:
    return ensure_dir(resolve_path("data/eval/downloaded"))


def cache_path(name: str) -> Path:
    spec = REGISTRY[name]
    return cache_dir() / f"{name}{Path(spec['filename']).suffix or '.txt'}"


def describe(name: str) -> str:
    spec = REGISTRY.get(name)
    return f"{name} — {spec['desc']}" if spec else name


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------
def _http_get(url: str, timeout: float = 30.0) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Word2Vec-lab/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:  # noqa: BLE001
        return None


def fetch(name: str, force: bool = False, quiet: bool = False) -> Optional[Path]:
    """下载并缓存数据集；成功返回文件路径，失败返回 None（不抛异常）。"""
    if name not in REGISTRY:
        return None
    dest = cache_path(name)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest

    spec = REGISTRY[name]
    for url in spec["urls"]:
        data = _http_get(url)
        if data and len(data) > 32:
            dest.write_bytes(data)
            if not quiet:
                print(f"  已下载 {name}: {len(data)} 字节  <- {url}")
            return dest
        if not quiet:
            print(f"  下载失败，换下一个源: {url}")

    # 主源全挂：尝试用可重建的子集拼回来
    for combo in (spec.get("rebuild_from"),):
        if not combo:
            continue
        parts = [fetch(part, force=force, quiet=quiet) for part in combo]
        if all(parts):
            text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in parts)
            dest.write_text(text, encoding="utf-8")
            if not quiet:
                print(f"  已用 {' + '.join(combo)} 重建 {name}")
            return dest

    if not quiet:
        print(f"  [警告] 无法下载 {name}，将回退到内置小评测集。")
        print(f"          可手动下载后放到: {dest}")
        for url in spec["urls"]:
            print(f"            {url}")
    return None


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------
def parse_similarity_text(text: str) -> List[Tuple[str, str, float]]:
    """解析词相似度文件。

    兼容三种写法：
        word1<TAB>word2<TAB>score      （word-sim 标准格式）
        word1 word2 score              （内置小评测集）
        Word 1,Word 2,Human (mean)     （CSV 带表头，会被自动跳过）
    """
    pairs: List[Tuple[str, str, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "\t" in line:
            fields = [f.strip() for f in line.split("\t") if f.strip()]
            if len(fields) < 3:
                continue
            w1, score_str, w2 = fields[0], fields[-1], fields[1]
        else:
            fields = line.replace(",", " ").split()
            if len(fields) < 3:
                continue
            # 空格分隔时，中间可能有多个字段（多词短语），最后一个才是分值
            w1, score_str, w2 = fields[0], fields[-1], " ".join(fields[1:-1])

        try:
            score = float(score_str)
        except ValueError:
            continue  # 表头行（"Word 1 Word 2 Human (mean)"）会走到这里

        w1, w2 = w1.lower(), w2.lower()
        if w1 and w2:
            pairs.append((w1, w2, score))
    return pairs


def parse_analogy_text(text: str) -> List[Tuple[str, str, str, str, str]]:
    """解析类比文件，返回 (a, b, c, d, category)。

    支持 Google Analogy 的 ": 分组名" 段头；没有段头时 category 为空串。
    """
    questions: List[Tuple[str, str, str, str, str]] = []
    category = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(":"):
            category = line.lstrip(": ").strip()
            continue
        parts = line.split()
        if len(parts) >= 4:
            questions.append((parts[0].lower(), parts[1].lower(),
                              parts[2].lower(), parts[3].lower(), category))
    return questions


# --------------------------------------------------------------------------
# 统一加载入口
# --------------------------------------------------------------------------
def _read(spec: str, bundled: str, fetch_enabled: bool,
          quiet: bool) -> Tuple[str, str, Optional[Path]]:
    """把「注册名」或「文件路径」统一解析成 (标签, 文本, 路径)。"""
    if is_registered(spec):
        path = fetch(spec, quiet=quiet) if fetch_enabled else cache_path(spec)
        if path and path.exists():
            return spec, path.read_text(encoding="utf-8", errors="ignore"), path
        if not fetch_enabled:
            print(f"  [提示] {spec} 尚未下载（可用 python -m src.eval_data --fetch {spec}）")
    else:
        path = resolve_path(spec)
        if path and path.exists():
            return Path(spec).name, path.read_text(encoding="utf-8", errors="ignore"), path
        print(f"  [警告] 找不到评测文件 {spec}")

    # 回退到内置小评测集
    fallback = resolve_path(bundled)
    if fallback and fallback.exists():
        return f"{Path(bundled).name}（内置兜底）", \
            fallback.read_text(encoding="utf-8", errors="ignore"), fallback
    return "none", "", None


def load_similarity(spec: str, fetch_enabled: bool = True,
                    quiet: bool = False) -> Tuple[str, List[Tuple[str, str, float]]]:
    label, text, _ = _read(spec, BUNDLED_SIMILARITY, fetch_enabled, quiet)
    return label, parse_similarity_text(text)


def load_analogy(spec: str, fetch_enabled: bool = True,
                 quiet: bool = False,
                 ) -> Tuple[str, List[Tuple[str, str, str, str, str]]]:
    label, text, _ = _read(spec, BUNDLED_ANALOGY, fetch_enabled, quiet)
    return label, parse_analogy_text(text)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    import argparse

    from .utils import force_utf8_stdout

    force_utf8_stdout()
    p = argparse.ArgumentParser(description="评测数据集下载 / 查看")
    p.add_argument("--fetch", nargs="*", default=None,
                   help="下载指定数据集（留空表示全部）")
    p.add_argument("--list", action="store_true", help="列出所有可下载数据集")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if a.list:
        print("可用数据集：")
        for name, spec in REGISTRY.items():
            cached = "已缓存" if cache_path(name).exists() else "未下载"
            print(f"  {name:16s} [{spec['kind']:10s}] {cached}  {spec['desc']}")
        return

    names = a.fetch if a.fetch else list(REGISTRY)
    if not names:
        return
    ok = 0
    for name in names:
        if not is_registered(name):
            print(f"未知数据集: {name}")
            continue
        path = fetch(name, force=a.force)
        if path:
            ok += 1
            print(f"  已就绪: {path}")
    print(f"\n完成 {ok}/{len(names)}。缓存目录: {cache_dir()}")


if __name__ == "__main__":
    main()
