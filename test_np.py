import numpy as np


# token_embedding × weight = 新 embedding
def embedding_weight():
    x = np.array([1, 2, 3])
    w = np.random.randn(3, 4)
    y = x @ w
    print("x:", x)
    print("w:", w)
    print("y:", y)


# 多个token
def embedding_weight_real():
    tokens = np.random.randn(5, 768)
    W = np.random.randn(768, 768)
    out = tokens @ W
    print(out)


# 模拟Transformer 模型（如 GPT、BERT）中“自注意力机制 (Self-Attention)”的第一步：线性投影
def linear(tokens):
    # 5: 代表一句话里有 5 个单词（Token）
    # 768: 每个单词的特征维度（这是 BERT-base 模型的标准维度）

    # Wq (Query 模具)：把原始 Token 塑造成“寻找者”。它只提取那些能表达“我想找什么”的特征。
    Wq = np.random.randn(768, 768)
    # Wk (Key 模具)：把原始 Token 塑造成“索引标签”。它只提取那些能表达“我是谁，我有什么属性”的特征。
    Wk = np.random.randn(768, 768)
    # Wv (Value 模具)：把原始 Token 塑造成“内容载体”。它提取的是最纯粹、最深刻的语义信息。
    Wv = np.random.randn(768, 768)
    # 通过矩阵乘法 @，原始的 tokens 向量被投影到了三个不同的“语义空间”：
    # Q (Query - 查询向量)：“我要找谁？”
    Q = tokens @ Wq
    # K (Key - 键向量)：“我是谁？”
    K = tokens @ Wk
    # V (Value - 值向量)：“我有什么？”
    V = tokens @ Wv
    return (Q, K, V)


# 计算每两个单词之间的匹配得分（Attention Score）
def lite_attention_score(Q, K):
    # K.T 是 K 的转置，形状变为 (768, 5)
    # (5, 768) @ (768, 5) = (5, 5)
    # ***结果矩阵中的每一个元素 Score_{i,j} 代表：第 i 个词对第 j 个词的关注度。***
    scores = Q @ K.T
    return scores


# 缩放点积 (Scaled Dot-Product)
# 直接使用 \(Q@K^{T}\) 会导致数值过大，从而使梯度消失。标准的 Attention 论文 建议除以特征维度的平方根 \(\sqrt{d_{k}}\)
def scaled_dot_attention_score(Q, K):
    # 获取Q的最后一个维度 (5, 768) 获取的就是 768
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    return scores


# 在 NumPy 中实现 Softmax 非常直观。它的作用是将 \(Q@K^{T}\) 算出来的那些杂乱的分数转换为概率分布（即所有数字都在 0 到 1 之间，且每一行的总和等于 1）
def softmax(x):
    # 1. 为了数值稳定性，减去每行的最大值 (防止 exp 结果太大导致内存溢出)
    # axis=-1 表示在最后一个维度（行）上操作，keepdims=True 保持维度以便广播
    # 计算e的“差值”次方 差值<=0 计算结果都在 (0, 1]之间
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # 2. 计算每一行的指数和
    row_sums = e_x.sum(axis=-1, keepdims=True)
    # 3. 归一化：每个元素除以所在行的总和
    attention_weight = e_x / row_sums
    return attention_weight


def attention_out(attn, V):
    # 4. 【核心一步】执行信息聚合
    # attn (5, 5) @ V (5, 768) = out (5, 768)
    return attn @ V


def test_attention_out():
    # 5 个单词（Token）
    tokens = np.random.randn(5, 768)
    # 线性投影
    (Q, K, V) = linear(tokens)
    # 缩放点积
    scores = scaled_dot_attention_score(Q, K)
    # Softmax
    attentions = softmax(scores)
    # 信息聚合
    out = attention_out(attentions, V)
    return out


if __name__ == "__main__":
    out = test_attention_out()
    print("out", out.shape, out)
