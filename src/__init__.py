"""Word2Vec-lab：一个从零实现、可复现的 Word2Vec 训练流程。

模块划分（对应流程的七个阶段）：
    collect     数据收集
    preprocess  数据清理 + 分词 + 词表 + ID 序列
    dataset     训练样本生成（正样本对 + 负采样 + 高频词下采样）
    model       模型框架（Skip-gram / CBOW + Negative Sampling）
    train       模型训练
    infer       模型推理
    evaluate    定量评估（相似度 / 类比）
    visualize   可视化（PCA / t-SNE / TensorBoard Projector）
"""

__version__ = "1.0.0"
