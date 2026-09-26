#!/usr/bin/env bash
# ============================================================
# 服务器一键跑完整流程（在仓库根目录执行）
#
#   bash scripts/run_server.sh              # 正式训练 text8
#   CONFIG=configs/tiny.yaml bash scripts/run_server.sh   # 小规模自测
#
# ⚠️ 训练会用到 RTX 5090。请先联系导师开启显卡，再运行本脚本。
#    脚本第 1 步会自动体检，没开卡会直接停下并提示。
# ============================================================
set -euo pipefail

CONFIG="${CONFIG:-configs/full.yaml}"
PY="${PY:-python}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " Word2Vec-lab  服务器流程"
echo " 配置: $CONFIG    时间: $(date '+%F %T')"
echo "=============================================================="

# ---------- 1. 显卡体检（必须通过） ----------
echo
echo "[1/7] 检查 GPU 环境 ..."
if ! "$PY" scripts/gpu_check.py; then
  cat <<'EOF'

  ============================================================
   GPU 体检未通过，流程中止。
   请先联系导师申请开启 RTX 5090，并按上面的提示逐项排查。
   只想先跑通流程、不占卡的话，把配置里的 train.device 改成 cpu：
       CONFIG=configs/tiny.yaml bash scripts/run_server.sh
  ============================================================
EOF
  exit 1
fi

# ---------- 2. 评测集（提前拉，免得训练完了才发现下不动） ----------
echo
echo "[2/7] 准备评测数据集 ..."
"$PY" -m src.eval_data --fetch 2>&1 | tee "$LOG_DIR/evaldata.log"

# ---------- 3. 语料 ----------
SOURCE=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['data']['source'])")
RUN=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['run_name'])")
SIM=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['eval']['similarity_file'])")
ANA=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['eval']['analogy_file'])")
echo
echo "[3/7] 收集语料 (source=$SOURCE) ..."
"$PY" -m src.collect --source "$SOURCE" 2>&1 | tee "$LOG_DIR/collect.log"

# ---------- 4. 预处理 ----------
echo
echo "[4/7] 数据清理与预处理 ..."
"$PY" -m src.preprocess --config "$CONFIG" 2>&1 | tee "$LOG_DIR/preprocess.log"

# ---------- 5. 样本自检 ----------
echo
echo "[5/7] 训练样本自检 ..."
"$PY" -m src.dataset --config "$CONFIG" --show 8 2>&1 | tee "$LOG_DIR/samples.log"

# ---------- 6. 训练 ----------
echo
echo "[6/7] 开始训练（时间预算由配置里的 train.max_minutes 控制，超时自动保存）..."
echo "      想断线不中断，请用 tmux："
echo "        tmux new -s w2v  ->  再执行本脚本  ->  Ctrl+B D 脱离"
"$PY" -m src.train --config "$CONFIG" --viz 2>&1 | tee "$LOG_DIR/train_$RUN.log"

# ---------- 7. 用完整数据集重新评估 ----------
echo
echo "[7/7] 独立评估 (相似度=$SIM, 类比=$ANA) ..."
"$PY" -m src.evaluate \
  --vectors "runs/$RUN/vectors.npz" \
  --similarity "$SIM" --analogy "$ANA" \
  --out "runs/$RUN/eval.json" 2>&1 | tee "$LOG_DIR/eval_$RUN.log"

cat <<EOF

==============================================================
 全部完成
   词向量 : runs/$RUN/vectors.npz / vectors.txt
   检查点 : runs/$RUN/last.pt
   指标   : runs/$RUN/eval.json
   可视化 : runs/$RUN/viz/{pca,tsne}.png
   日志   : $LOG_DIR/
==============================================================
EOF
