from src.model.encoder import TrajectoryEncoder
from src.model.decoder import TrajectoryDecoder
import torch.nn as nn
import torch

class TrajectoryTransformer(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        nhead_encoder: int = 8,
        nhead_decoder: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        dropout: float = 0.1,
        coord_dim: int = 2,
        traj_input_dim: int = 2,
        freeze_backbone: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.coord_dim = coord_dim

        self.encoder = TrajectoryEncoder(
            d_model=d_model,
            nhead=nhead_encoder,
            n_layers=n_encoder_layers,
            dropout=dropout,
            traj_input_dim=traj_input_dim,
            freeze_backbone=freeze_backbone,
            max_seq_len=max_seq_len,
        )

        self.decoder = TrajectoryDecoder(
            d_model=d_model,
            nhead=nhead_decoder,
            n_layers=n_decoder_layers,
            dropout=dropout,
            coord_dim=coord_dim,
            max_seq_len=max_seq_len,
        )

    @staticmethod
    def _lengths_to_pad_mask(
        lengths: torch.Tensor, max_len: int, device: torch.device,
    ):
        arange = torch.arange(max_len, device=device).unsqueeze(0)
        return arange >= lengths.unsqueeze(1)

    def forward(
        self,
        frames: torch.Tensor,
        tgt: torch.Tensor,
        device: torch.device,
        frame_lengths: torch.Tensor | None = None,
        tgt_lengths: torch.Tensor | None = None,
        encoder_trajectory: torch.Tensor | None = None,
        encoder_traj_lengths: torch.Tensor | None = None,
    ):
        B, T_f = frames.shape[0], frames.shape[1]
        T_tgt = tgt.shape[1]

        frame_pad_mask = (
            self._lengths_to_pad_mask(frame_lengths, T_f, device)
            if frame_lengths is not None else None
        )

        encoder_traj_pad_mask = None
        if encoder_trajectory is not None and encoder_traj_lengths is not None:
            T_t = encoder_trajectory.shape[1]
            encoder_traj_pad_mask = self._lengths_to_pad_mask(
                encoder_traj_lengths, T_t, device,
            )

        tgt_pad_mask = (
            self._lengths_to_pad_mask(tgt_lengths, T_tgt, device)
            if tgt_lengths is not None else None
        )

        memory = self.encoder(
            frames=frames,
            device=device,
            trajectory=encoder_trajectory,
            frame_pad_mask=frame_pad_mask,
            traj_pad_mask=encoder_traj_pad_mask,
        )

        memory_pad_mask = self._build_memory_pad_mask(
            T_f, frame_pad_mask,
            encoder_trajectory, encoder_traj_pad_mask,
            B, device,
        )

        coords, stop_logits = self.decoder(
            tgt=tgt,
            memory=memory,
            device=device,
            tgt_pad_mask=tgt_pad_mask,
            memory_pad_mask=memory_pad_mask,
        )

        return coords, stop_logits

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        device: torch.device,
        max_steps: int = 100,
        stop_threshold: float = 0.5,
        start_pos: torch.Tensor | None = None,
        encoder_trajectory: torch.Tensor | None = None,
        frame_lengths: torch.Tensor | None = None,
        encoder_traj_lengths: torch.Tensor | None = None,
    ):
        if start_pos is None:
            if encoder_trajectory is None:
                raise ValueError(
                    "Either start_pos or encoder_trajectory must be provided"
                )
            if encoder_traj_lengths is not None:
                last_idx = (encoder_traj_lengths - 1).long()
                start_pos = encoder_trajectory[
                    torch.arange(encoder_trajectory.shape[0], device=device), last_idx
                ]
            else:
                start_pos = encoder_trajectory[:, -1, :]
        self.eval()
        B, T_f = frames.shape[0], frames.shape[1]

        frame_pad_mask = (
            self._lengths_to_pad_mask(frame_lengths, T_f, device)
            if frame_lengths is not None else None
        )

        encoder_traj_pad_mask = None
        if encoder_trajectory is not None and encoder_traj_lengths is not None:
            T_t = encoder_trajectory.shape[1]
            encoder_traj_pad_mask = self._lengths_to_pad_mask(
                encoder_traj_lengths, T_t, device,
            )

        memory = self.encoder(
            frames=frames,
            device=device,
            trajectory=encoder_trajectory,
            frame_pad_mask=frame_pad_mask,
            traj_pad_mask=encoder_traj_pad_mask,
        )

        memory_pad_mask = self._build_memory_pad_mask(
            T_f, frame_pad_mask,
            encoder_trajectory, encoder_traj_pad_mask,
            B, device,
        )

        generated = start_pos.unsqueeze(1)
        all_stop_probs = []

        for _ in range(max_steps):
            coords, stop_logits = self.decoder(
                tgt=generated,
                memory=memory,
                device=device,
                memory_pad_mask=memory_pad_mask,
            )

            next_coord = coords[:, -1:, :]
            stop_prob = torch.sigmoid(stop_logits[:, -1:, :])

            generated = torch.cat([generated, next_coord], dim=1)
            all_stop_probs.append(stop_prob)

            if (stop_prob.squeeze(-1) > stop_threshold).all():
                break

        all_coords = generated[:, 1:, :]
        all_stop_probs = torch.cat(all_stop_probs, dim=1)
        return all_coords, all_stop_probs

    @staticmethod
    def _build_memory_pad_mask(
        T_f: int,
        frame_pad_mask: torch.BoolTensor | None,
        encoder_trajectory: torch.Tensor | None,
        encoder_traj_pad_mask: torch.BoolTensor | None,
        B: int,
        device: torch.device,
    ):
        
        if encoder_trajectory is not None:
            T_t = encoder_trajectory.shape[1]
            if frame_pad_mask is None:
                frame_pad_mask = torch.zeros(B, T_f, dtype=torch.bool, device=device)
            if encoder_traj_pad_mask is None:
                encoder_traj_pad_mask = torch.zeros(B, T_t, dtype=torch.bool, device=device)
            return torch.cat([frame_pad_mask, encoder_traj_pad_mask], dim=1)

        return frame_pad_mask