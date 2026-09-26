"""通用工具：配置加载、随机种子、日志、设备解析（含 GPU 授权提醒）、计时。"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: Optional[str]) -> Optional[Path]:
    """相对路径统一按项目根目录解析，避免 cwd 不同导致找不到文件。"""
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def ensure_dir(p) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取 YAML 配置，并允许用点号路径覆盖，例如 {"train.lr": 0.01}。"""
    if yaml is None:
        raise ImportError("需要 pyyaml：pip install pyyaml")
    cfg_path = resolve_path(path)
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def save_json(obj: Any, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# 随机性
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------
def force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，中文输出会花屏或抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def get_logger(name: str = "word2vec", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复添加 handler
        return logger
    force_utf8_stdout()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fp = Path(log_file)
        ensure_dir(fp.parent)
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------
# GPU 提醒 / 设备解析
# --------------------------------------------------------------------------
GPU_REMINDER = """
======================================================================
 需要开启 GPU（RTX 5090）—— 请先联系导师
======================================================================
 本步骤在 GPU 上训练可以快 30~100 倍，但 5090 目前处于未启用状态。
 请向导师申请：
   1) 开通 5090 的使用权限（账号加入对应分组 / 等前面的人释放显卡）；
   2) 确认服务器 NVIDIA 驱动 >= 570，以支持 CUDA 12.8；
   3) 确认 PyTorch 装的是 cu128 版本（sm_120 架构必需）。

 验证命令：
   nvidia-smi
   python -c "import torch;print(torch.cuda.get_device_name(0))"

 暂时没开卡也可以先用 CPU 把流程跑通： --set train.device=cpu
======================================================================
"""


def print_gpu_reminder() -> None:
    print(GPU_REMINDER, flush=True)


def resolve_device(spec: str = "auto", logger: Optional[logging.Logger] = None):
    """把配置里的 device 字符串解析成 torch.device。

    - 请求 cuda 但不可用 -> 打印提醒并抛出明确异常（而不是静默退回 CPU）
    - auto -> 有卡用卡，没卡用 CPU
    """
    import torch

    log = logger.info if logger else print
    spec = (spec or "auto").lower()

    available = torch.cuda.is_available()
    if spec in ("cuda", "gpu") and not available:
        print_gpu_reminder()
        raise RuntimeError(
            "请求使用 CUDA 但当前不可用：卡可能还没开，或驱动 / PyTorch 版本不匹配。"
            "请按上面的提示联系导师开卡；临时可改用 --set train.device=cpu"
        )

    if spec == "cpu":
        log("使用设备: CPU（提示：正式训练请申请开启 5090）")
        return torch.device("cpu")

    if available:
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        log(f"使用设备: cuda:0  {name}  sm_{cc[0]}{cc[1]}  显存 {total:.1f} GB")
        log("提醒：请确认已获得导师授权使用该显卡。")
        return torch.device("cuda")

    log("未检测到可用 CUDA，自动回退 CPU。正式训练请申请开启 5090。")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 计时与格式化
# --------------------------------------------------------------------------
class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.marks: Dict[str, float] = {}

    def mark(self, name: str) -> float:
        dt = time.time() - self.t0
        self.marks[name] = dt
        return dt

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0


def format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def human_int(n: float) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))
