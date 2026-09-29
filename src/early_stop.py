"""早停：从评估结果里取监控指标，并判断是否该停。

刻意不依赖 PyTorch —— 这样 `tests/test_pipeline.py` 可以脱离训练框架单独验证它，
也让「什么指标、耐心多少」这件事和模型实现解耦。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 监控指标别名 -> 规范名
METRIC_ALIASES = {
    "analogy": "analogy_accuracy",
    "analogy_accuracy": "analogy_accuracy",
    "similarity": "similarity_spearman",
    "similarity_spearman": "similarity_spearman",
    "mean": "mean",
    "combined": "mean",
}

METRIC_DESC = {
    "analogy_accuracy": "Google Analogy 准确率（a/b/c 都在词表内的题）",
    "similarity_spearman": "WordSim-353 的 Spearman 秩相关",
    "mean": "上述两者的平均（都是 0~1 量纲）",
}


def canonical_metric(monitor: str) -> str:
    return METRIC_ALIASES.get(monitor, monitor)


def is_finite(v: Any) -> bool:
    """None / nan / inf 都算「取不到」。"""
    return v is not None and bool(np.isfinite(v))


def extract_metric(res: Dict, monitor: str) -> Optional[float]:
    """从一次评估结果里取出用于早停的标量，指标越大越好；取不到返回 None。"""
    key = canonical_metric(monitor)
    ana = (res.get("analogy") or {}).get("accuracy")
    sim = (res.get("similarity") or {}).get("spearman")

    if key == "analogy_accuracy":
        return float(ana) if is_finite(ana) else None
    if key == "similarity_spearman":
        return float(sim) if is_finite(sim) else None
    if key == "mean":
        vals = [float(v) for v in (ana, sim) if is_finite(v)]
        return float(np.mean(vals)) if vals else None
    return None


class EarlyStopper:
    """按监控指标做早停，并记录最优 epoch。

    规则（指标一律「越大越好」）：
      * 比历史最优高出超过 min_delta 才算一次提升，重置耐心计数；
      * 连续 patience 次评估都没有提升，且已跑够 min_epochs 个 epoch 就停；
      * 某次评估取不到指标（例如所有题都缺词）时既不记为提升、也不消耗耐心。
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0,
                 min_epochs: int = 1) -> None:
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.min_epochs = max(0, int(min_epochs))
        self.best: Optional[float] = None
        self.best_epoch: int = -1
        self.bad_epochs: int = 0
        self.n_evals: int = 0
        self.history: List[Tuple[int, Optional[float]]] = []

    def is_better(self, value: Optional[float]) -> bool:
        if not is_finite(value):
            return False
        if self.best is None:
            return True
        return float(value) > self.best + self.min_delta

    def step(self, epoch: int, value: Optional[float]) -> bool:
        """喂入一次评估结果（epoch 从 0 开始），返回 True 表示应当停止训练。"""
        self.n_evals += 1
        self.history.append((epoch, None if not is_finite(value) else float(value)))

        if not is_finite(value):
            return False

        if self.is_better(value):
            self.best, self.best_epoch, self.bad_epochs = float(value), epoch, 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience and (epoch + 1) >= self.min_epochs

    def summary(self) -> str:
        if self.best is None:
            return "尚未得到有效指标"
        return f"最优 epoch {self.best_epoch + 1}（{self.best:.4f}）"

    def to_dict(self, monitor: str = "", stopped: bool = False) -> Dict:
        """落盘用：整条指标曲线也存下来，方便画图看早停是否合理。"""
        return {
            "enabled": True,
            "monitor": canonical_metric(monitor),
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "best_epoch": self.best_epoch,
            "best_epoch_display": self.best_epoch + 1,
            "best_value": self.best,
            "n_evals": self.n_evals,
            "stopped_by_early_stopping": bool(stopped),
            "history": [{"epoch": e, "epoch_display": e + 1, "metric": v}
                        for e, v in self.history],
        }
