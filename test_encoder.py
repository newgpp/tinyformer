import torch
from model.encoder import EncoderConfig, Encoder

torch.manual_seed(0)

cfg = EncoderConfig(
    d_model=32,
    n_heads=4,
    d_ff=128,
    n_layers=2,
    dropout=0.0
)

enc = Encoder(cfg)

B, T = 2, 6
x = torch.randn(B, T, cfg.d_model)

# 假设最后两个位置是 padding
kpm = torch.zeros(B, T, dtype=torch.bool)
kpm[:, -2:] = True

y = enc(x, key_padding_mask=kpm)
print("encoder out:", y.shape)
