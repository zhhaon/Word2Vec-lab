"""服务器 GPU 体检脚本 —— 开卡之前/之后都建议先跑一遍。

    python scripts/gpu_check.py

重点检查 RTX 5090（Blackwell, sm_120）的三大前提：
    1) 驱动版本足够新（>= 570，支持 CUDA 12.8）
    2) PyTorch 是 cu128 版本
    3) 该 PyTorch 版本内置了 sm_120 的 kernel（torch >= 2.7）
"""
from __future__ import annotations

import platform
import sys


def line(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    line("0. 基本环境")
    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"平台        : {platform.platform()}")

    line("1. 检查 PyTorch")
    try:
        import torch
    except ImportError:
        print("[X] 未安装 PyTorch。")
        print("    安装命令（Blackwell 必需 cu128）：")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1

    print(f"torch       : {torch.__version__}")
    print(f"编译时 CUDA : {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")
    arch_list = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    print(f"内置架构    : {arch_list or '（未检测到 CUDA，无法读取）'}")

    if torch.version.cuda is None:
        print("[X] 这是 CPU 版 PyTorch，无法使用显卡。")
        print("    请重装 cu128 版本（见 README「服务器部署」）。")
        return 1

    line("2. 检查 CUDA 可用性")
    if not torch.cuda.is_available():
        print("[X] torch.cuda.is_available() = False")
        print("    常见原因：")
        print("      a) 显卡还没开 —— 请联系导师申请开启 RTX 5090")
        print("      b) 驱动过旧     —— nvidia-smi 看 Driver Version，需 >= 570")
        print("      c) 装成了 CPU 版 —— 见上面第 1 步")
        print("      d) 容器没挂显卡 —— 确认启动时加了 --gpus all")
        return 1

    n = torch.cuda.device_count()
    print(f"[OK] 检测到 {n} 张显卡")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        cc = torch.cuda.get_device_capability(i)
        print(f"  cuda:{i}  {p.name}  sm_{cc[0]}{cc[1]}  "
              f"{p.total_memory / 1024 ** 3:.1f} GB  "
              f"MP={p.multi_processor_count}")

    line("3. 检查 sm_120 kernel 是否可用（5090 关键项）")
    cc = torch.cuda.get_device_capability(0)
    sm = f"sm_{cc[0]}{cc[1]}"
    has_kernel = any(sm in a for a in arch_list) or "*" in "".join(arch_list)
    if not has_kernel:
        print(f"[!] 当前 PyTorch 的 kernel 列表里没有 {sm}")
        print(f"    列表: {arch_list}")
        print("    5090 需要 torch >= 2.7 + cu128。请升级后重试。")
    else:
        print(f"[OK] 找到 {sm} 对应 kernel")

    line("4. 实跑一个小算子")
    try:
        dev = torch.device("cuda:0")
        a = torch.randn(4096, 4096, device=dev)
        b = torch.randn(4096, 4096, device=dev)
        c = a @ b
        torch.cuda.synchronize()
        print(f"[OK] 矩阵乘法通过，结果范数 = {float(c.norm()):.2f}")

        emb = torch.nn.Embedding(100000, 300, sparse=True, device=dev)
        idx = torch.randint(0, 100000, (8192,), device=dev)
        out = emb(idx).sum()
        out.backward()
        torch.cuda.synchronize()
        grad = emb.weight.grad
        print(f"[OK] 稀疏梯度正常，grad 为 sparse={grad.is_sparse}，nnz={grad._nnz()}")

        opt = torch.optim.SparseAdam(emb.parameters(), lr=1e-3)
        opt.step()
        print("[OK] SparseAdam 可用")
    except Exception as exc:  # noqa: BLE001
        print(f"[X] 算子执行失败: {type(exc).__name__}: {exc}")
        print("    如果报 'no kernel image is available'，说明 torch/CUDA 版本与 5090 不匹配。")
        return 1

    line("结论")
    print("[OK] 环境就绪，可以开始正式训练：")
    print("     python -m src.collect   --source text8")
    print("     python -m src.preprocess --source text8 --min-count 5")
    print("     python -m src.train     --config configs/full.yaml --viz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
