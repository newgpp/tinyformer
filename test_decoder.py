import torch
from model.decoder import DecoderConfig, Decoder

torch.manual_seed(0)

cfg = DecoderConfig(
    d_model=32,
    n_heads=4,
    d_ff=128,
    n_layers=2,
    dropout=0.0
)
dec = Decoder(cfg)

B, T, S = 2, 5, 7
x = torch.randn(B, T, cfg.d_model)       # decoder 输入（一般是 embedding+pos）
mem = torch.randn(B, S, cfg.d_model)     # encoder 输出

# encoder 的最后两位是 padding
mem_kpm = torch.zeros(B, S, dtype=torch.bool)
mem_kpm[:, -2:] = True

y = dec(x, mem, memory_key_padding_mask=mem_kpm)
print("decoder out:", y.shape)
