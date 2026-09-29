# Word2Vec-lab

一个**从零实现、全流程可复现**的 Word2Vec 教学项目，覆盖数据收集 → 数据清理 → 训练样本生成 → 模型框架 → 模型训练 → 模型推理 → 可视化 七个阶段。

- 模型部分**不调用任何现成词向量库**，Skip-gram / CBOW + Negative Sampling 全部手写（约 150 行 PyTorch），每一行都能对应到论文公式。
- 两档配置：`tiny`（1 分钟跑完，本地 CPU 即可）与 `full`（text8 语料 + RTX 5090，约 10~25 分钟）。
- 正式评估用**完整的公开标准数据集**：WordSim-353（353 行）+ Google Analogy Dataset（19544 题 / 14 类），
  指标可直接与论文对比；**中途评估用的是同一套集**，所以 epoch 之间的数字可比。
- 内置**早停**：按监控指标自动停，并保留最优 epoch 的权重与词向量。
- 训练全程有**时间预算保护**，硬上限 6 小时以内，超时自动优雅停止并保存。

---

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 快速开始（本地，CPU，~2 分钟）](#2-快速开始本地cpu2-分钟)
- [3. 七个阶段详解](#3-七个阶段详解)
- [4. 服务器部署（RTX 5090）](#4-服务器部署rtx-5090)
- [5. 代码同步到 GitHub](#5-代码同步到-github)
- [6. 结果解读与调参](#6-结果解读与调参)
- [7. 常见问题](#7-常见问题)

---

## 1. 环境准备

```
Word2Vec-lab/
├── configs/          # tiny / full 两档配置
├── data/
│   ├── samples/      # 内置离线语料与可视化词表（入库）
│   ├── eval/         # 内置小评测集（入库）
│   │   └── downloaded/  # 标准数据集缓存：WordSim-353 / Google Analogy（不入库）
│   ├── raw/          # 下载的原始语料（不入库）
│   └── processed/    # 预处理结果（不入库）
├── src/
│   ├── collect.py     # ① 数据收集
│   ├── preprocess.py  # ② 数据清理 + 分词 + 词表
│   ├── dataset.py     # ③ 训练样本生成（向量化）
│   ├── model.py       # ④ 模型框架
│   ├── train.py       # ⑤ 模型训练
│   ├── infer.py       # ⑥ 模型推理
│   ├── visualize.py   # ⑦ 可视化
│   ├── evaluate.py    #    定量评估（相似度 / 3CosAdd 类比）
│   ├── eval_data.py   #    标准评测集下载、缓存与解析
│   ├── early_stop.py  #    早停判定（不依赖 PyTorch，可单独测试）
│   ├── vectors.py     #    词向量存取与运算
│   ├── vocab.py       #    词表 / 负采样表
│   └── demo_corpus.py #    离线语料生成器
├── scripts/          # GPU 体检 / 服务器一键流程 / Slurm 模板
├── tests/            # 无需 PyTorch 的自检脚本
└── Makefile
```

依赖见 `requirements.txt`。**注意 PyTorch 的安装方式**（5090 是 Blackwell 架构，必须 cu128）：

```bash
# 服务器（CUDA 12.8 / RTX 50 系）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 本地 CPU 调试（Python 3.8 最高只能装 torch 2.4）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 2. 快速开始（本地，CPU，~2 分钟）

先确认整套代码是对的，再上服务器：

```bash
make check          # 环境体检（服务器上必跑）
make local          # 全流程：生成语料 -> 预处理 -> 训练 12 epoch -> 可视化
python tests/test_pipeline.py   # 35 项自检，不需要 PyTorch
```

`make local` 结束后会得到：

```
runs/tiny/
├── vectors.npz / vectors.txt   # 词向量（早停开启时 = 最优 epoch 的那份）
├── best_vectors.npz / .txt     # 最优 epoch 的向量
├── best.pt / best_meta.json    # 最优权重 + 它对应哪个 epoch、指标多少
├── eval.json                   # 最终评估（相似度 / 类比）
├── early_stopping.json         # 每个 epoch 的指标曲线 + 是否因早停而停
├── metrics.csv                 # 逐步 loss / lr / 吞吐
├── last.pt                     # 检查点（可续训）
├── train.log
└── viz/
    ├── pca.png / tsne.png      # 语义聚类图
    ├── loss_curve.png
    └── projector/              # 可拖进 TensorBoard Projector
```

预期指标（demo 语料，只用于验证流程是否正常；配置是每 2 个 epoch 评估一次）：

| 指标 | 典型值 | 说明 |
|---|---|---|
| 类比准确率 | ≈ 0.68~0.78 | `king:queen :: man:woman` 这类 |
| 相似度 Spearman | ≈ 0.28~0.33 | 偏低是 Word2Vec 各向异性的正常现象，见 [第 6 节](#6-结果解读与调参) |

这个数据集上 `mean`（两者平均）大约在 epoch 13~17 见顶，早停会据此决定何时收工。

---

## 3. 七个阶段详解

### ① 数据收集 `src/collect.py`

```bash
python -m src.collect --list                    # 看有哪些数据源
python -m src.collect --source demo             # 离线小语料（内置生成器，~6 万词）
python -m src.collect --source wikitext2        # WikiText-2，~200 万词
python -m src.collect --source text8            # 经典基准，~1700 万词【推荐】
python -m src.collect --source local --local-path /path/to/your.txt
```

统一产出 `data/raw/<name>/raw.txt`。text8 网络不通的话手动下载 `http://mattmahoney.net/dc/text8.zip` 放到 `data/raw/text8/` 下再重跑。

### ② 数据清理 `src/preprocess.py`

```bash
python -m src.preprocess --config configs/full.yaml
```

做的事：去掉控制字符 / 零宽字符 → 正则分词（英文 `[a-z]+('...)?`，中文走 jieba）→ 词频统计 → 建词表（低于 `min_count` 的词并入 `<unk>`）→ 编码成 `int32` ID 序列。

**关键设计**：保留句子边界（`sent_lengths.npy`），训练时窗口不会跨句乱连。全部流式分块处理，1700 万词语料内存占用 < 1 GB。

### ③ 训练样本生成 `src/dataset.py`

```bash
python -m src.dataset --config configs/full.yaml --show 20   # 肉眼检查样本长什么样
```

这是整套流程里最容易被写慢的一环。1700 万词的一个 epoch 要产出约 1 亿个样本对，纯 Python 循环会成为绝对瓶颈。这里用**全向量化的 numpy** 生成，速度提升两个数量级。

同时实现论文里的两个关键技巧：

- **动态窗口**：每个中心词的窗口半径 `r ~ Uniform{1..W}`，离得近的词被采到更多次；
- **高频词下采样**：`P(保留) = sqrt(t/f) + t/f`，削弱 `the/of` 的支配地位。

```3:5:src/dataset.py
本模块用**全向量化的 numpy** 生成样本对，而不是 Python 逐词循环。
```

> ⚠️ `t` 必须随语料规模缩放。`t=1e-4` 是 1e8 词级语料的经验值，用在几万词的小语料上会把 `the` 砍到只剩 4%。

### ④ 模型框架 `src/model.py`

Skip-gram：

$$J = -\log \sigma(u_o \cdot v_c) - \sum_{k=1}^{K} \log \sigma(-u_k \cdot v_c)$$

CBOW：把上下文向量取平均得到 $v_{ctx}$，其余相同。代码里几个**实际训练才会踩到的坑**：

- `Embedding(sparse=True)`：本项目 text8 的 V=7.1 万、D=300，稠密梯度每步要写
  2140 万个 float（86 MB）；稀疏梯度只更新 batch 里出现的几千行，快得多。
  配套必须用 `SparseAdam`（AdamW 不支持稀疏梯度）。
- 正样本与 K 个负样本拼在一起做一次 `BCEWithLogits`，等价于原式且数值更稳。
- 损失按 `sum / batch_size` 聚合（对 1+K 项求和后按样本平均），与主流实现一致。
- 梯度裁剪自己实现，因为 `torch.nn.utils.clip_grad_norm_` 对稀疏梯度不友好。

#### 负采样的两条硬约束

`src/vocab.py::NegativeSampler.sample_excluding()` 保证每一条负样本都不触犯下面两条：

**1. 排除本样本的 positive target（假负样本）**

同一行里 label=1 的那个词，绝不能再被抽成这一行的负样本 —— 否则就是一边说
「`king → queen` 是真共现」，一边又把 `queen` 当负样本往 $\sigma(-u_{queen}\cdot v_{king}) \to 0$ 推，
自己在跟自己的标签打架。

排除范围按结构区分（`src/train.py::sample_negatives`）：

| 结构 | positive target | 额外排除 | 理由 |
|---|---|---|---|
| Skip-gram | 上下文词 `context[b]` | 中心词 `center[b]` | 中心词与正目标是一对真共现，同属假负样本 |
| CBOW | 中心词 `center[b]` | 整条上下文 `context[b, :]` | 上下文本身就是正例的输入，拿来当负例同样自相矛盾 |

**2. 排除 `<unk>` 与 `<pad>`**

`<unk>` / `<pad>` 不携带语义，被抽中只是白送噪声，还会让 `<unk>` 的向量被大量无意义梯度污染。
实现上有两道防线：

- 构建 unigram^0.75 采样表时把它们的词频清零（分不到任何槽位），并在建表后断言表里不含它们；
- 采样时再显式过滤一次，这样即使以后换了别的采样表实现，约束也不会被悄悄破坏。

**实现方式**是拒绝采样：冲突概率约 $K/V$（text8 的 V=7.1 万、K=10 时约 1.4e-4），正常一轮就结束；
词表极小的极端情况有确定性兜底。为了不让每次采样都触发 GPU→CPU 同步，
负采样放在**预取线程**里用 numpy 完成，与 GPU 计算重叠。

训练日志会打印实际冲突率，可以直接确认规则生效：

```
负采样约束: 排除本样本的正目标(上下文词) + 排除中心词 + 排除 <unk>/<pad>
负采样统计: 共 4.62M 个，其中与正目标/特殊符号冲突而被重抽 86.35K 个（1.8701%）
```

### ⑤ 模型训练 `src/train.py`

```bash
python -m src.train --config configs/full.yaml --viz
python -m src.train --config configs/full.yaml --resume runs/full/last.pt   # 续训
python -m src.train --config configs/full.yaml --set train.lr=0.001 --set model.dim=200
```

- 后台线程预取数据，GPU 不空转；
- 线性学习率衰减（原论文做法）；
- TensorBoard + `metrics.csv` 双路日志；
- 定期保存检查点，支持断点续训；
- **完整评估 + 早停**（见下）；
- **时间预算保护** `train.max_minutes`：到点自动停止并保存，保证不超 6 小时。

#### 中途评估与早停

每到 `eval.every_epochs` 个 epoch 结束，就用**完整的评测集**评估一次当前模型：

```
[epoch 5 评估] 相似度 rho=0.6421（覆盖 99%）  |  类比 acc=0.4532（18104/19544 题）
            语义=0.5102  句法=0.4021
            监控 analogy_accuracy = 0.4532   最优: 最优 epoch 5（0.4532）   未提升 0/3
            指标提升，已保存最优权重 -> runs/full/best.pt
```

早停规则（`early_stopping` 段配置，指标一律「越大越好」）：

| 配置项 | 含义 |
|---|---|
| `monitor` | `analogy_accuracy`（默认）/ `similarity_spearman` / `mean` |
| `patience` | 连续多少次评估没超过最优值就停 |
| `min_delta` | 提升幅度小于它不算提升（full 的 19544 题下 0.002 ≈ 39 题） |
| `min_epochs` | 至少训练几个 epoch 才允许停 |
| `export_best` | 用最优 epoch 的权重覆盖 `vectors.*`（默认 true） |

两条容易踩的细节，实现里都处理了：

- **指标算不出来时（`nan`）不消耗耐心**。语料太小时所有题都可能因缺词而无法评估，
  这时如果算作「未提升」，会在完全没信息的情况下把训练停掉；
- **`test_pipeline.py` 覆盖了早停逻辑**（提升重置耐心、`min_delta` 门槛、`min_epochs`
  保护、`nan` 不污染计数），共 25 项断言。

`monitor` 怎么选：`analogy_accuracy` 是 Word2Vec 最标准的对比指标，粒度也够细；
语料很小（比如内置集只有 41 题）时它一格就是 0.024，抖动大，这时用 `mean` 更稳。

产物（早停开启时）：

```
runs/full/
├── vectors.npz / .txt        # export_best=true 时 = 最优 epoch 的向量
├── best_vectors.npz / .txt   # 最优 epoch 的向量（单独留一份）
├── best.pt                   # 最优 epoch 的权重
├── best_meta.json            # 最优是哪个 epoch、指标多少
├── last.pt                   # 最后一个 epoch 的状态（续训用）
├── early_stopping.json       # 完整的指标曲线 + 是否因早停而停
├── eval_epoch003.json ...    # 每次中途评估的完整结果
└── eval.json                 # 最终评估结果
```

### ⑥ 模型推理 `src/infer.py`

```bash
python -m src.infer --vectors runs/full/vectors.npz                       # 交互模式
python -m src.infer --vectors runs/full/vectors.npz -c "ana man king woman"
python -m src.infer --vectors runs/full/vectors.npz -c "nn paris 10"
python -m src.infer --vectors runs/full/vectors.npz -c "arith king - man + woman"
```

交互模式里输入 `help` 看全部命令。输入拼错的词会自动给出拼写建议。

定量评估 `src/evaluate.py`：词相似度 Spearman 相关 + 3CosAdd 类比准确率。

```bash
# tiny：用内置小评测集（离线、秒级）
python -m src.evaluate --vectors runs/tiny/vectors.npz \
    --similarity data/eval/similarity_pairs.txt \
    --analogy data/eval/analogy_questions.txt

# full：用完整的公开标准数据集（首次自动下载并缓存）
python -m src.evaluate --vectors runs/full/vectors.npz \
    --similarity wordsim353 --analogy google-analogy \
    --out runs/full/eval.json
```

#### 评测集

评测集参数既可以是 **文件路径**，也可以是 **注册名**。注册名会按需下载并缓存到
`data/eval/downloaded/`，下载失败自动回退到内置小评测集（不会中断流程）：

```bash
python -m src.eval_data --list        # 列出所有可选数据集及缓存状态
python -m src.eval_data --fetch       # 提前把全部数据集拉下来（推荐训练开始前跑）
make evaldata                         # 同上
```

| 注册名 | 规模 | 说明 |
|---|---|---|
| `wordsim353` | 353 行 / 352 唯一词对 | WordSim-353 全量，**full 配置默认** |
| `wordsim353-sim` | 203 对 | 相似性子集 |
| `wordsim353-rel` | 252 对 | 相关性自集 |
| `men` | 3000 对 | MEN |
| `simlex999` | 999 对 | SimLex-999 |
| `google-analogy` | 19544 题 / 14 类 | Google Analogy Dataset，**full 配置默认** |

关于 WordSim-353 有一个容易让人以为解析写错了的事实：**它的文件有 353 行，但
`(money, cash)` 出现了两次**（9.15 与 9.08），所以唯一词对是 352 个。代码保留全部行、
另外报告唯一词对数，不做静默去重。报告里会明确打印这一点。

类比评测按 Google 官方的 14 个类别分组，并汇总成论文里的两个大组：

- **semantic**（语义，8869 题）：capital-common-countries / capital-world / currency / city-in-state / family
- **syntactic**（句法，10675 题）：gram1 ~ gram9

同时给出两个口径，避免误读：

- `accuracy` = 答对 / **a、b、c 三词都在词表里的题数**（论文常用口径，可与文献对比）
- `accuracy_total` = 答对 / 全部题（缺词的题算错，最严格）

> 实现上不是逐题搜索，而是把所有题的 query 向量一次性算出来再分块做矩阵乘。
> 逐题算的话 19544 题 × 7.1 万词表要十几分钟；分块 GEMM 让 BLAS 吃满，
> text8 的规模下约 40 秒。分块大小由 `chunk_mb` 控制（默认 128 MB）。

### ⑦ 可视化 `src/visualize.py`

```bash
python -m src.visualize --vectors runs/tiny/vectors.npz \
    --words-file data/samples/demo_words.txt --methods pca tsne \
    --metrics runs/tiny/metrics.csv
```

- PCA / t-SNE / UMAP 三种降维，`demo_words.txt` 里带类别，画出来会**按语义簇上色**；
- 导出 TensorBoard Projector 需要的 `vectors.tsv` + `metadata.tsv`：

```bash
tensorboard --logdir runs/full/tb --port 6006
# 浏览器打开 http://localhost:6006/#projector，上传 viz/projector 下的两个 tsv
```

---

## 4. 服务器部署（RTX 5090）

### 4.0 先联系导师开卡

本项目需要用到 5090，但**当前显卡处于未启用状态**。在跑任何训练之前：

> **请联系导师申请：**
> 1. 开通 5090 的使用权限（账号加入对应分组 / 等前面的人释放卡）；
> 2. 确认服务器 NVIDIA 驱动 **≥ 570**（支持 CUDA 12.8）；
> 3. 确认 PyTorch 装的是 **cu128** 版本。

代码里内置了提醒：只要配置里写了 `train.device: cuda` 而卡不可用，训练会**立刻停下并打印上面的提示**，不会静默退回 CPU 让你白等。

### 4.1 登录服务器并体检

```bash
ssh your_name@lab-server
nvidia-smi                      # 看有没有卡、驱动版本、显存占用
```

```bash
git clone <你的仓库地址> Word2Vec-lab && cd Word2Vec-lab
python scripts/gpu_check.py     # 逐项体检，重点检查 sm_120 kernel 是否存在
```

`gpu_check.py` 会依次验证：Python/torch 版本 → CUDA 可用性 → **sm_120 kernel 是否存在**（5090 关键项）→ 实跑矩阵乘 + 稀疏梯度 + `SparseAdam`。任何一项不过都会给出具体原因。

### 4.2 环境安装

```bash
conda create -n w2v python=3.11 -y
conda activate w2v                  # 别装 Python 3.8，torch 2.7+ 装不上

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

验证：

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# 期望输出类似: 2.7.1+cu128 12.8 NVIDIA GeForce RTX 5090
```

### 4.3 跑全流程

```bash
bash scripts/run_server.sh                       # 默认 configs/full.yaml
CONFIG=configs/tiny.yaml bash scripts/run_server.sh   # 小规模先验一遍
```

脚本会依次执行：GPU 体检 → **下载评测集** → 数据收集 → 预处理 → 样本自检 → 训练 →
用完整数据集评估，日志落在 `logs/`。

评测集提前单独拉一次也很方便（失败会立刻暴露，而不是等训练跑完）：

```bash
make evaldata        # 或 python -m src.eval_data --fetch
make datasets        # 查看缓存状态
```

**防止 SSH 断连中断训练**：

```bash
tmux new -s w2v        # 建会话
bash scripts/run_server.sh
# Ctrl+B 然后按 D 脱离；tmux attach -t w2v 回来看进度
```

或后台跑：

```bash
nohup bash scripts/run_server.sh > logs/run.out 2>&1 &
tail -f logs/run.out
```

集群用 Slurm 的话直接改 `scripts/train.sbatch` 里的队列名后 `sbatch scripts/train.sbatch`。

### 4.4 时间预算

先看一个容易被搞错的点：**这个流程的瓶颈不是 5090，而是数据管道。**

负采样放在预取线程里用 numpy 完成（为了不让每步都触发 GPU→CPU 同步），
实测「PairStream + 负采样」的吞吐约 **86 万样本对/秒**；
而这个模型小到 5090 算它绰绰有余。所以训练速度由 CPU 侧的样本生成决定。

`configs/full.yaml` 的估算（text8，1700 万词，dim=300，5 epoch）：

真实 text8 上实测的规模（`python -m src.dataset --config configs/full.yaml --show 5` 随时可复现）：

| 指标 | 实测值 |
|---|---|
| 语料 token | 17,005,207 |
| 词表（min_count=5） | 71,292（`<unk>` 占 1.68%） |
| subsample=1e-4 后 | 9.67M token（保留 56.9%） |
| 每 epoch 样本对 | 58.04M |
| 模型参数 | 2 × 71292 × 300 ≈ 4280 万 |

| 阶段 | 时间 | 说明 |
|---|---|---|
| 下载 text8 | 取决于网速 | 31 MB 压缩包；慢的话本地下好再 scp |
| 下载评测集 | < 10 秒 | `make evaldata`，约 600 KB |
| 预处理 | **约 45 秒** | 实测：1700 万词两遍流式遍历 |
| 训练 | **9~17 分钟** | 单 epoch ≈ 70 秒（数据管道上限），跑满 8 个 epoch ≈ 9 分钟 |
| 中途评估 | **约 1 分钟 / 次** | 每 epoch 一次（实测 56 秒），8 次约 8 分钟；**早停通常会砍掉几次** |
| 最终评估 + 可视化 | 2~4 分钟 | |

早停开启的情况下，实际训练时长通常比上表更短——这正是它的作用。
但注意评估是**同步**做的（不占 GPU，占 CPU 单核），所以每个 epoch 的总开销
是「训练 + 1 分钟评估」。真嫌评估慢可以把 `eval.every_epochs` 调成 2。

中途评估与最终评估用的是**同一套**评测集（都由配置里的 `eval.similarity_file` /
`eval.analogy_file` 决定），所以 full 跑起来每个 epoch 都会过一次完整的
WordSim-353 + Google Analogy。这不只是为了好看指标——**只有同一套题，epoch 之间的
数字才可比，早停才有依据**。

单次评估约 1 分钟，8 个 epoch 全跑满的评估开销约 8 分钟，相对训练本身（约 9 分钟）
是可以接受的；反过来它换来的是「哪一轮最好」这个信息，以及早停省下的时间。

配置里设了 `max_minutes: 330`（5.5 小时）作为兜底，任何情况下都会在 6 小时内结束并保存。
真嫌慢的话，加大 `train.prefetch`（预取深度）能让 GPU 少等 CPU，但没法突破管道本身的吞吐上限。

### 4.5 取回结果

训练产物被 `.gitignore` 排除了，别用 git 传大文件：

```bash
# 本地执行
scp -r your_name@lab-server:~/Word2Vec-lab/runs/full/viz ./full_viz
scp your_name@lab-server:~/Word2Vec-lab/runs/full/eval.json ./full_eval.json
```

词向量本体（`vectors.npz`，text8 + dim300 约 300 MB）按需用 `rsync -avz --progress` 拉回。

---

## 5. 代码同步到 GitHub

### 5.1 首次上传（本地）

```bash
cd Word2Vec-lab
git init
git add .
git commit -m "feat: Word2Vec 全流程（数据/样本/模型/训练/推理/可视化）"

# 在 GitHub 上新建一个空仓库（不要勾选 README / .gitignore），然后：
git branch -M main
git remote add origin git@github.com:<你的用户名>/Word2Vec-lab.git
git push -u origin main
```

没有配 SSH key 就用 HTTPS：

```bash
git remote add origin https://github.com/<你的用户名>/Word2Vec-lab.git
git push -u origin main
```

### 5.2 已经确认不会上传的东西

`.gitignore` 已排除：`data/raw/`、`data/processed/`、`data/eval/downloaded/`、`runs/`、
`*.pt`、`*.npy`、`*.npz`、`logs/`。
**会**上传的是 `data/samples/` 与 `data/eval/*.txt`（几 KB 的离线语料和内置评测集），
这样 clone 下来 `make local` 就能直接跑。

标准评测集（WordSim-353 / Google Analogy，共约 600 KB）不入库，用
`make evaldata` 重新下载即可；服务器无外网的话在本地下载后 scp 过去：

```bash
python -m src.eval_data --fetch          # 本地下载到 data/eval/downloaded/
scp -r data/eval/downloaded 服务器:~/Word2Vec-lab/data/eval/
```

### 5.3 服务器上拉取 / 更新

```bash
# 服务器首次
git clone git@github.com:<你的用户名>/Word2Vec-lab.git && cd Word2Vec-lab

# 本地改完代码后
git add -A && git commit -m "..." && git push
# 服务器上
git pull
```

服务器上如果 clone 时用 HTTPS 每次都要输密码，建议配 SSH key：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub        # 复制到 GitHub -> Settings -> SSH and GPG keys
ssh -T git@github.com            # 测试
```

### 5.4 提交前建议跑一遍

```bash
python tests/test_pipeline.py
```

---

## 6. 结果解读与调参

### 怎么判断训练是否正常

1. **loss 起点**应为 $(1+K)\ln 2$。K=5 → ≈ 4.16，K=10 → ≈ 7.62。如果不是这个数，说明目标函数或损失缩放写错了。
2. **loss 应平滑下降**到 1.3~2.5 区间（取决于语料大小与 K）。
3. `gnorm` 不应持续发散了再被你裁回来。
4. **Google Analogy 准确率**：text8 + skip-gram + dim300 上 **≈ 0.5~0.70**；
   其中 semantic 组通常明显高于 syntactic 组。demo 语料上约 0.68（但那套题是围绕 demo 语料写的）。
5. **WordSim-353 Spearman**：text8 + dim300 上 **≈ 0.6~0.75**。
   demo 语料上只有 8% 的词对可评估，看覆盖率就知道这个数字不该拿来对比。

看评估输出时先看**覆盖率**：`--similarity` 报的 coverage 和 `--analogy` 报的
`n_evaluated / n_questions`。覆盖率太低说明语料/词表太小，指标本身没有对比意义。

### 「训练越久，类比越好但相似度越差」是正常的

Word2Vec 学出的向量空间是**各向异性**的：所有词都挤在一个窄锥里，余弦相似度普遍偏高（0.5~0.9），于是相似度评测的秩相关被压扁。这是被反复观察到的现象（*all-but-the-top* 那篇论文研究的正是它）。

所以：

- **相似度指标偏低不代表向量没用**，看近邻和类比更直观；
- 想改善可以试：语料换成自然文本（text8 而不是模板生成语料）、调小 `train.epochs`、或对向量做去均值 / 去主成分后处理。

### 主要超参

| 参数 | 建议 | 影响 |
|---|---|---|
| `model.dim` | 100（小语料）/ 300（大语料） | 太小欠拟合，太大在小语料上过拟合 |
| `model.window` | 5 | 小窗口偏「功能相似」，大窗口偏「主题相似」 |
| `model.negative` | 5~15 | 越大越慢，收敛更稳 |
| `model.subsample` | 大语料 1e-4 / 小语料 1e-2 或 0 | 见 [③](#-训练样本生成-srcdatasetpy) |
| `model.arch` | skipgram | skipgram 对低频词更好；cbow 快几倍 |
| `train.lr` | SparseAdam 用 2.5e-3~5e-3 | Adam 家族比原始 SGD 的 0.025 小一个量级 |
| `train.optimizer` | sparseadam | 想复现原论文可换 `sgd` + `lr=0.025` |
| `train.batch_size` | 8192 | 只影响吞吐，不太影响效果 |
| `early_stopping.monitor` | `analogy_accuracy` | 语料小、题目少时改用 `mean` 更稳 |
| `early_stopping.patience` | 3 | 越大越保守；配合 `eval.every_epochs` 一起看 |
| `eval.every_epochs` | 1~2 | 越小早停越灵敏，但评估开销越高 |

---

## 7. 常见问题

**Q: 报 `no kernel image is available for execution on the device`**
PyTorch 版本没带 sm_120 kernel。`pip install torch --index-url https://download.pytorch.org/whl/cu128`，且版本 ≥ 2.7。

**Q: 报 `CUDA out of memory`**
本项目的显存占用主要是参数的 4~5 倍（参数 + 梯度 + 优化器状态）。`dim=300`、词表 7.1 万时约 0.7 GB，5090 的 32 GB 绰绰有余。真爆了就把 `train.batch_size` 减半。

**Q: 报 `AdamW does not support sparse gradients`**
`model.sparse=true` 必须配 `sparseadam` / `adagrad` / `sgd`。想用 AdamW 就把 `model.sparse` 设为 `false`（会慢一些、显存多占一些）。

**Q: 训练中途断了怎么办**
`python -m src.train --config configs/full.yaml --resume runs/full/last.pt`，会从上次检查点继续。

**Q: 想换自己的语料**
```bash
python -m src.collect --source local --local-path /path/to/corpus.txt
# 然后在配置里把 data.source 改成 local
python -m src.preprocess --config configs/full.yaml
```

**Q: 想跑中文**
在 `configs/full.yaml` 里设 `data.lang: zh`（会自动用 jieba 分词），并在 `src/collect.py` 的 `SOURCES` 里加上你的中文语料来源；或直接用 `--source local` 喂中文维基 dump。

**Q: Windows 控制台中文乱码**
命令行前执行 `chcp 65001`，或改用 Windows Terminal。代码里已经尝试把 stdout 切到 UTF-8。
