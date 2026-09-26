"""第 7 步：可视化。

产出：
    1) pca.png / tsne.png     —— 词向量降到 2D 后散点展示语义聚类
    2) projector/             —— TensorBoard Projector 需要的 vectors.tsv + metadata.tsv
    3) loss_curve.png         —— 训练损失曲线（读 metrics.csv）

用法：
    python -m src.visualize --vectors runs/full/vectors.npz
    python -m src.visualize --vectors runs/full/vectors.npz \
        --words-file data/samples/demo_words.txt --methods pca tsne
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import ensure_dir, get_logger, human_int, resolve_path
from .vectors import EmbeddingMatrix, load_vectors


# --------------------------------------------------------------------------
# 选词
# --------------------------------------------------------------------------
def read_words_file(path) -> Tuple[List[str], Dict[str, str]]:
    """读取词表文件；支持 `word` 或 `word category` 两种写法。"""
    words: List[str] = []
    categories: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            words.append(parts[0])
            if len(parts) > 1:
                categories[parts[0]] = parts[1]
    return words, categories


def select_words(emb: EmbeddingMatrix, words_file: Optional[str],
                 top_words: int) -> Tuple[List[str], Dict[str, str]]:
    if words_file:
        p = resolve_path(words_file)
        if p and p.exists():
            words, cats = read_words_file(p)
            words = [w for w in words if w in emb]
            return words, {w: c for w, c in cats.items() if w in emb}
        print(f"  [警告] 找不到词表文件 {words_file}，改用高频词")

    # 词向量按词频降序存储，跳过 <unk>/<pad> 即为最高频词
    words = [w for w in emb.words[:top_words + 2]
             if not (w.startswith("<") and w.endswith(">"))][:top_words]
    return words, {}


# --------------------------------------------------------------------------
# 降维
# --------------------------------------------------------------------------
def reduce_pca(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :2] * S[:2]).astype(np.float32)


def reduce_tsne(X: np.ndarray, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE

    perp = float(min(perplexity, max(5.0, (len(X) - 1) / 3.0)))
    common = dict(n_components=2, perplexity=perp, init="pca", random_state=seed)
    try:
        # sklearn >= 1.2
        tsne = TSNE(learning_rate="auto", **common)
    except TypeError:
        tsne = TSNE(learning_rate=200.0, **common)
    return tsne.fit_transform(X).astype(np.float32)


def reduce_umap(X: np.ndarray, seed: int = 42) -> np.ndarray:
    import umap  # 需要 pip install umap-learn

    return umap.UMAP(n_components=2, random_state=seed).fit_transform(X).astype(np.float32)


REDUCERS = {"pca": reduce_pca, "tsne": reduce_tsne, "umap": reduce_umap}


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------
def plot_scatter(coords: np.ndarray, words: Sequence[str], categories: Dict[str, str],
                 title: str, out_path: Path, annotate_limit: int = 300) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

    if categories:
        cats = sorted(set(categories.values()))
        cmap = plt.get_cmap("tab20")
        color_of = {c: cmap(i % 20) for i, c in enumerate(cats)}
        for c in cats:
            idx = [i for i, w in enumerate(words) if categories.get(w) == c]
            if not idx:
                continue
            ax.scatter(coords[idx, 0], coords[idx, 1], s=28, alpha=0.85,
                       color=color_of[c], label=f"{c} ({len(idx)})")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=22, alpha=0.7, color="#2b6cb0")

    if len(words) <= annotate_limit:
        for i, w in enumerate(words):
            ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=8,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    if categories and len(cats) <= 20:
        ax.legend(loc="best", fontsize=8, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss(metrics_csv: Path, out_path: Path) -> Optional[Path]:
    if not metrics_csv.exists():
        return None
    steps, losses = [], []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
            except (KeyError, ValueError):
                continue
    if not steps:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(steps, losses, color="#c53030", linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Word2Vec training loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_projector(emb: EmbeddingMatrix, words: Sequence[str], out_dir: Path) -> None:
    """导出 TensorBoard Projector 所需文件：

        tensorboard --logdir <out_dir>/../tb --port 6006
    然后打开 http://localhost:6006/#projector 上传这两个 tsv。
    """
    out_dir = ensure_dir(out_dir)
    idx = [emb.stoi[w] for w in words]
    with open(out_dir / "vectors.tsv", "w", encoding="utf-8") as f:
        for i in idx:
            f.write("\t".join(f"{v:.6f}" for v in emb.matrix[i]) + "\n")
    with open(out_dir / "metadata.tsv", "w", encoding="utf-8") as f:
        f.write("word\n")
        for w in words:
            f.write(w + "\n")


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def visualize(vectors_path, out_dir=None, words_file=None, top_words: int = 300,
              methods: Sequence[str] = ("pca", "tsne"), tsne_perplexity: float = 30.0,
              seed: int = 42, metrics_csv=None, logger=None) -> Path:
    log = logger.info if logger else print
    emb = load_vectors(vectors_path)
    log("载入词向量: %s 词, %d 维", human_int(len(emb)), emb.dim)

    words, categories = select_words(emb, words_file, top_words)
    if len(words) < 5:
        raise ValueError("可可视化的词太少（<5），请检查词表文件或 top_words")
    log("待可视化词数: %d%s", len(words),
        f"（{len(set(categories.values()))} 个类别）" if categories else "")

    out_dir = ensure_dir(resolve_path(out_dir) if out_dir
                         else Path(vectors_path).parent / "viz")
    X = emb.subset(words).matrix

    for method in methods:
        method = method.lower()
        if method not in REDUCERS:
            log("跳过未知降维方法: %s", method)
            continue
        log("降维中: %s ...", method)
        try:
            coords = REDUCERS[method](X, tsne_perplexity, seed) if method == "tsne" \
                else REDUCERS[method](X)
        except ImportError as exc:
            log("  %s 不可用（%s），跳过", method, exc)
            continue
        out_png = out_dir / f"{method}.png"
        plot_scatter(coords, words, categories,
                     f"Word2Vec embeddings — {method.upper()} "
                     f"({len(words)} words, dim={emb.dim})", out_png)
        log("  已保存 %s", out_png)

    export_projector(emb, words, out_dir / "projector")
    log("TensorBoard Projector 文件已导出: %s", out_dir / "projector")

    if metrics_csv:
        p = plot_loss(resolve_path(metrics_csv), out_dir / "loss_curve.png")
        if p:
            log("损失曲线已保存: %s", p)

    return out_dir


def visualize_run(cfg: Dict, vectors_path=None, logger=None) -> Path:
    """供 train.py 在训练结束后自动调用。"""
    v = cfg.get("viz", {}) or {}
    t = cfg.get("train", {}) or {}
    run_name = cfg.get("run_name", "run")
    run_dir = resolve_path(f"{t.get('out_dir', 'runs')}/{run_name}")

    return visualize(
        vectors_path=vectors_path or (run_dir / "vectors.npz"),
        out_dir=v.get("out_dir") or (run_dir / "viz"),
        words_file=v.get("words_file"),
        top_words=int(v.get("top_words", 300)),
        methods=v.get("methods", ["pca", "tsne"]),
        tsne_perplexity=float(v.get("tsne_perplexity", 30.0)),
        seed=int(cfg.get("seed", 42)),
        metrics_csv=str(run_dir / "metrics.csv"),
        logger=logger,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="词向量可视化")
    p.add_argument("--vectors", required=True, help="vectors.npz")
    p.add_argument("--out", default=None, help="输出目录，默认 <vectors 同级的 viz/>")
    p.add_argument("--words-file", default=None,
                   help="要画的词表（每行 word [category]）；不给则取高频词")
    p.add_argument("--top-words", type=int, default=300)
    p.add_argument("--methods", nargs="+", default=["pca", "tsne"],
                   choices=["pca", "tsne", "umap"])
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--metrics", default=None, help="metrics.csv 路径，用于画损失曲线")
    a = p.parse_args()

    visualize(a.vectors, a.out, a.words_file, a.top_words, a.methods,
              a.tsne_perplexity, metrics_csv=a.metrics, logger=get_logger("word2vec"))


if __name__ == "__main__":
    main()
