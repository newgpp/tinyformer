import torch
from model.attention import AttentionConfig, MultiHeadAttention

torch.manual_seed(0)

cfg = AttentionConfig(d_model=32, n_heads=4, dropout=0.0)
mha = MultiHeadAttention(cfg)

B, T = 2, 5
x = torch.randn(B, T, cfg.d_model)

# 1) Self-attention (encoder style)
y, attn = mha(x, return_attn=True)
print("self:", y.shape, attn.shape)

# 2) Causal self-attention (decoder style)
y2, attn2 = mha(x, causal=True, return_attn=True)
print("causal:", y2.shape, attn2.shape)

# 3) Cross-attention
mem = torch.randn(B, 7, cfg.d_model)
y3, attn3 = mha(x, memory=mem, return_attn=True)
print("cross:", y3.shape, attn3.shape)

# 4) Key padding mask example (mask last 2 keys in memory)
kpm = torch.zeros(B, mem.shape[1], dtype=torch.bool)
kpm[:, -2:] = True
y4, attn4 = mha(x, memory=mem, key_padding_mask=kpm, return_attn=True)
print("cross+padmask:", y4.shape, attn4.shape)
