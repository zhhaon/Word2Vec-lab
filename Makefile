# ============================================================
# Word2Vec-lab 常用命令
#
# 本地快速跑通（CPU，几分钟）：   make local
# 服务器正式训练（5090）：        make data && make preprocess && make train
# ============================================================
PY ?= python
CONFIG ?= configs/full.yaml
SOURCE ?= $(shell $(PY) -c "import yaml;print(yaml.safe_load(open('$(CONFIG)'))['data']['source'])")
RUN ?= $(shell $(PY) -c "import yaml;print(yaml.safe_load(open('$(CONFIG)'))['run_name'])")

.PHONY: help check install data preprocess preview train resume eval infer viz local all clean-dist

help:
	@echo "make check         # GPU / 环境体检（服务器上先跑这个）"
	@echo "make install       # 安装依赖（torch 需先按 README 单独装 cu128 版）"
	@echo "make data          # 下载语料（SOURCE=text8 / wikitext2 / demo）"
	@echo "make preprocess    # 清洗 + 分词 + 词表 + ID 序列"
	@echo "make preview       # 打印生成的训练样本，肉眼检查"
	@echo "make train         # 训练（CONFIG=configs/full.yaml）"
	@echo "make resume        # 从 runs/<RUN>/last.pt 续训"
	@echo "make eval          # 评估（相似度 + 类比）"
	@echo "make infer         # 交互式推理"
	@echo "make viz           # 出可视化图"
	@echo "make local         # 端到端跑一遍 tiny 配置（CPU）"
	@echo "make all           # 数据 -> 预处理 -> 训练 -> 评估 -> 可视化"

check:
	$(PY) scripts/gpu_check.py

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.collect --source $(SOURCE)

preprocess:
	$(PY) -m src.preprocess --config $(CONFIG)

preview:
	$(PY) -m src.dataset --config $(CONFIG) --show 20

train:
	$(PY) -m src.train --config $(CONFIG) --viz

resume:
	$(PY) -m src.train --config $(CONFIG) --resume runs/$(RUN)/last.pt

eval:
	$(PY) -m src.evaluate --vectors runs/$(RUN)/vectors.npz --out runs/$(RUN)/eval.json

infer:
	$(PY) -m src.infer --vectors runs/$(RUN)/vectors.npz

viz:
	$(PY) -m src.visualize --vectors runs/$(RUN)/vectors.npz \
		--words-file "$(WORDS_FILE)" --methods pca tsne \
		--metrics runs/$(RUN)/metrics.csv

local:
	$(PY) -m src.collect    --source demo
	$(PY) -m src.preprocess --source demo --min-count 3 --max-tokens 200000
	$(PY) -m src.train      --config configs/tiny.yaml --set train.device=cpu --viz

all: data preprocess train eval viz

clean-dist:
	rm -rf data/processed runs
