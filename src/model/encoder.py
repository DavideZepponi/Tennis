from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import resnet18, ResNet18_Weights
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

    def forward(self, seq_len: int):
        return self.pe[:, :seq_len, :]

class VisualBackbone(nn.Module):

    def __init__(self, freeze: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = 512  # ResNet-18 last-layer channels

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def normalize(self, x: torch.Tensor):
        return (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)

    def forward(self, frames: torch.Tensor):
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        x = self.normalize(x)
        x = self.backbone(x)
        x = x.reshape(B, T, self.feat_dim)
        return x

class ModalityEmbedding(nn.Module):

    def __init__(self, n_modalities: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(n_modalities, d_model)

    def forward(self, modality_id: int, seq_len: int, device: torch.device):
        ids = torch.full((1, seq_len), modality_id, dtype=torch.long, device=device)
        return self.embedding(ids)

class TrajectoryEncoder(nn.Module):

    MODALITY_VIDEO = 0
    MODALITY_TRAJ = 1

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        traj_input_dim: int = 2,
        freeze_backbone: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        # ---- Visual ----
        self.visual_backbone = VisualBackbone(freeze=freeze_backbone)
        self.proj_visual = nn.Linear(self.visual_backbone.feat_dim, d_model)

        # ---- Trajectory ----
        self.proj_traj = nn.Linear(traj_input_dim, d_model)

        # ---- Positional & modality encoding ----
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.modality_embedding = ModalityEmbedding(n_modalities=2, d_model=d_model)

        # ---- Transformer encoder ----
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        # ---- Memory ----
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "visual_backbone" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        frames: torch.Tensor,
        device: torch.device,
        trajectory: torch.Tensor | None = None,
        frame_pad_mask: torch.BoolTensor | None = None,
        traj_pad_mask: torch.BoolTensor | None = None,
    ):
        B = frames.shape[0]

        # --- Visual ---
        vis_feat = self.visual_backbone(frames)
        vis_tokens = self.proj_visual(vis_feat)
        T_f = vis_tokens.shape[1]

        # --- Positional + modality for visual ---
        vis_tokens = (
            vis_tokens
            + self.pos_encoding(T_f)
            + self.modality_embedding(self.MODALITY_VIDEO, T_f, device)
        )

        # --- Trajectory ---
        if trajectory is not None:
            traj_tokens = self.proj_traj(trajectory)
            T_t = traj_tokens.shape[1]

            traj_pos = self.pos_encoding(T_f + T_t)[:, T_f:, :]
            traj_tokens = (
                traj_tokens
                + traj_pos
                + self.modality_embedding(self.MODALITY_TRAJ, T_t, device)
            )

            tokens = torch.cat([vis_tokens, traj_tokens], dim=1)

            if frame_pad_mask is not None or traj_pad_mask is not None:
                if frame_pad_mask is None:
                    frame_pad_mask = torch.zeros(B, T_f, dtype=torch.bool, device=device)
                if traj_pad_mask is None:
                    traj_pad_mask = torch.zeros(B, T_t, dtype=torch.bool, device=device)
                pad_mask = torch.cat([frame_pad_mask, traj_pad_mask], dim=1)
            else:
                pad_mask = None
        else:
            tokens = vis_tokens
            pad_mask = frame_pad_mask

        # --- Encode ---
        tokens = self.layer_norm(tokens)
        enc_out = self.encoder(tokens, src_key_padding_mask=pad_mask)
        memory = self.out_proj(enc_out)

        return memory