from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
import torch
import math

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len, :]

class TrajectoryDecoder(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        coord_dim: int = 2,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        # ---- Input projection ----
        self.proj_in = nn.Linear(coord_dim, d_model)

        # ---- Positional encoding ----
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.layer_norm = nn.LayerNorm(d_model)

        # ---- Transformer decoder ----
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=n_layers)

        # ---- Output head ----
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, coord_dim),
        )

        # ---- Stop head ----
        self.stop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        device: torch.device,
        tgt_pad_mask: torch.BoolTensor | None = None,
        memory_pad_mask: torch.BoolTensor | None = None,
    ):
        T_tgt = tgt.shape[1]

        # Project coordinates to d_model
        tgt_tokens = self.proj_in(tgt)

        # Add positional encoding
        tgt_tokens = tgt_tokens + self.pos_encoding(T_tgt)

        # Pre-LN
        tgt_tokens = self.layer_norm(tgt_tokens)

        # Causal mask
        causal_mask = self._causal_mask(T_tgt, device)

        # Decode
        dec_out = self.decoder(
            tgt=tgt_tokens,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )

        # Project to output coordinates
        coords = self.output_head(dec_out)

        # Stop probability
        stop_logits = self.stop_head(dec_out)

        return coords, stop_logits