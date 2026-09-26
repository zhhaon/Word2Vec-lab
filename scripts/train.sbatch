#!/usr/bin/env bash
# ============================================================
# Slurm 集群提交模板（很多高校实验室用 Slurm 管卡）
#
#   sbatch scripts/train.sbatch
#   squeue -u $USER          # 看排队状态
#   tail -f logs/slurm-%j.out
#
# 申请显卡前请先确认：导师已给你开通 5090 所在队列的使用权限。
# ============================================================
#SBATCH --job-name=word2vec
#SBATCH --partition=gpu            # ← 改成你们实验室的 GPU 队列名
#SBATCH --gres=gpu:1               # 申请 1 张卡
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00            # 硬上限 6 小时（与配置里的 max_minutes 呼应）
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

mkdir -p logs

# 如有需要，换成你们集群的环境激活方式：
#   source /etc/profile.d/conda.sh && conda activate w2v
#   module load cuda/12.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate w2v

cd "$SLURM_SUBMIT_DIR"

echo "节点: $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-未分配}"
nvidia-smi

python scripts/gpu_check.py

# 评测集提前拉，免得训练跑完才发现下不动（无外网就本地下好再 scp 过来）
python -m src.eval_data --fetch

SIM=$(python -c "import yaml;print(yaml.safe_load(open('configs/full.yaml'))['eval']['similarity_file'])")
ANA=$(python -c "import yaml;print(yaml.safe_load(open('configs/full.yaml'))['eval']['analogy_file'])")

python -m src.collect    --source text8
python -m src.preprocess --config configs/full.yaml
python -m src.train      --config configs/full.yaml --viz
python -m src.evaluate   --vectors runs/full/vectors.npz \
                         --similarity "$SIM" --analogy "$ANA" \
                         --out runs/full/eval.json

echo "完成: $(date '+%F %T')"
