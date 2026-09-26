"""第 4 步：模型框架。

从零实现两个 Word2Vec 变体（均使用 Negative Sampling 目标）：

Skip-gram
    J = -log σ(u_o · v_c) - Σ_{k=1..K} log σ(-u_k · v_c)
    输入侧矩阵 W_in (V×D) 就是我们要的词向量。

CBOW
    把上下文向量取平均得到 v_ctx，其余与 Skip-gram 相同。

实现细节说明（都是实际训练里的坑）：
    * Embedding 使用 sparse=True：本项目 text8 的 V≈7.1 万、D=300，稠密梯度
      每步要写 2140 万个 float（86MB），稀疏梯度只更新 batch 里出现的几千行，
      快得多；因此配套使用 torch.optim.SparseAdam（AdamW 不支持稀疏梯度）。
    * 正样本与 K 个负样本拼在一起做一次 BCEWithLogits，等价于原式但更省事、数值更稳。
    * 初始化 U(-0.5/D, 0.5/D)，与原论文 / gensim 一致。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .vocab import PAD_IDX


class Word2VecNeg(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 300,
        arch: str = "skipgram",
        pad_idx: int = PAD_IDX,
        sparse: bool = True,
    ) -> None:
        super().__init__()
        assert arch in ("skipgram", "cbow")
        self.vocab_size = vocab_size
        self.dim = dim
        self.arch = arch
        self.pad_idx = pad_idx

        # 输入 embedding：训练结束后导出它作为词向量
        self.in_embed = nn.Embedding(vocab_size, dim, sparse=sparse,
                                     padding_idx=None)
        # 输出 embedding：负采样目标里的 u
        self.out_embed = nn.Embedding(vocab_size, dim, sparse=sparse)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 0.5 / self.dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.zeros_(self.out_embed.weight)

    # ---------------- 前向 ----------------
    def _input_vector(self, center: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.arch == "skipgram":
            return self.in_embed(center)                       # (B, D)

        ctx = self.in_embed(context)                           # (B, 2W, D)
        mask = (context != self.pad_idx).unsqueeze(-1).to(ctx.dtype)   # (B, 2W, 1)
        summed = (ctx * mask).sum(dim=1)                       # (B, D)
        count = mask.sum(dim=1).clamp(min=1.0)                 # (B, 1)
        return summed / count

    def forward(
        self,
        center: torch.Tensor,
        context: torch.Tensor,
        negatives: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 logits (B, 1+K) 与 labels (B, 1+K)。

        negatives 形状：skipgram 为 (B, K)；cbow 为 (B, 1, K) 或 (B, K)。
        """
        v = self._input_vector(center, context)                # (B, D)
        u_pos = self.out_embed(context if self.arch == "skipgram"
                               else center)                    # (B, D)

        pos_logits = (v * u_pos).sum(dim=-1, keepdim=True)      # (B, 1)

        if negatives.dim() == 3:
            negatives = negatives.squeeze(1)
        u_neg = self.out_embed(negatives)                       # (B, K, D)
        neg_logits = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)   # (B, K)

        logits = torch.cat([pos_logits, neg_logits], dim=1)      # (B, 1+K)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1.0
        return logits, labels

    # ---------------- 导出 ----------------
    @torch.no_grad()
    def export_vectors(self, how: str = "input") -> torch.Tensor:
        """导出词向量矩阵 (V, D)。

        how:
            input  只取输入 embedding（标准做法，推荐）
            sum    输入 + 输出（部分任务上效果略好）
        """
        w_in = self.in_embed.weight.detach()
        if how == "input":
            return w_in.clone()
        if how == "sum":
            return (w_in + self.out_embed.weight.detach()).clone()
        raise ValueError(f"未知导出方式: {how}")

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float = 0.0):
    """按配置构建优化器；稀疏梯度必须用 SparseAdam。"""
    name = name.lower()
    params = model.parameters()

    if name == "sparseadam":
        return torch.optim.SparseAdam(params, lr=lr)
    if name in ("adamw", "adam"):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"未知优化器: {name}")


def lr_at_step(step: int, total_steps: int, base_lr: float, min_lr: float,
               warmup_frac: float = 0.0) -> float:
    """线性衰减（可选线性 warmup）。原论文就是「初始 lr 线性降到接近 0」。"""
    if total_steps <= 0:
        return base_lr
    warmup = max(1, int(total_steps * warmup_frac)) if warmup_frac > 0 else 0
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr + (min_lr - base_lr) * progress


def estimate_parameters(vocab_size: int, dim: int) -> str:
    n = 2 * vocab_size * dim
    return f"{n / 1e6:.1f}M 参数（约 {n * 4 / 1024 ** 2:.0f} MB fp32）"
