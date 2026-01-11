# tinyformer --- DEV_NOTES

本项目是一个从零手写的 Encoder--Decoder
Transformer，用于彻底理解注意力、投影、多头、cross-attention
等核心机制。 目标不是 SOTA，而是"可解释的、可推导的 Transformer"。

## 当前状态（已完成）

### Attention（model/attention.py）

-   支持 Self-Attention、Causal Self-Attention、Cross-Attention
-   支持 Padding Mask 与 Causal Mask
-   每个 head 是 R\^d_model -\> R\^d_head 的线性投影子空间

### Encoder（model/encoder.py）

结构（Pre-LN）： x -\> LN -\> Self-Attention -\> +x -\> LN -\> FFN -\>
+x\
输出 memory: \[B, S, d_model\]

### Decoder（model/decoder.py）

结构（Pre-LN）： x -\> LN -\> Causal Self-Attn -\> +x -\> LN -\>
Cross-Attn -\> +x -\> LN -\> FFN -\> +x\
Cross-attention 矩阵形状: \[B, n_heads, T, S\]

### Transformer Glue（model/transformer.py）

-   token embedding
-   learned positional embedding
-   encoder / decoder
-   LM head（与 tgt embedding 共享权重）
-   padding mask
-   forward(src, tgt, labels) -\> logits, loss

已验证 logits shape = \[B, T, vocab\]，loss 为正常浮点数。

## 已理解的核心概念

-   Token 是离散符号
-   Tokenizer = 符号系统
-   Embedding = 符号到向量
-   d_model = token 表示维度
-   Multi-head = 多投影子空间
-   Q/K/V = 查询 / 键 / 内容
-   Attention = 在投影空间里做相似度图

## 下一步 TODO

1.  data/copy_task.py 生成 seq2seq 复制任务
2.  train.py 训练循环（teacher forcing）
3.  验证 loss 下降
4.  可选：可视化 cross-attention 对齐

## 当前结论

已经成功从 0 搭出一个真实可运行的 Encoder--Decoder Transformer。
下一步是让它真正学会"复写"。
