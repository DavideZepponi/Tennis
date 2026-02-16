from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms.v2 as T
from tqdm.auto import tqdm
from pathlib import Path
import random
import torch
import math

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path: Path, preload: bool = False):
        self.dataset_path = path
        self.index = torch.load(path / "context" / "index.pt")
        self.targets_index = torch.load(path / "targets_frames" / "index.pt")
        self._metrics = self._load_or_compute_metrics()

        self._cache = {}
        if preload:
            for idx in tqdm(range(len(self.index)), desc="Preloading dataset"):
                self._cache[idx] = torch.load(self.index[idx]["file"])

    def _load_sample(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        return torch.load(self.index[idx]["file"])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sample = self._load_sample(idx)

        return (
            sample["id"],
            sample["X_frame"],
            self.standardize_coords(sample["X_coord"]),
            self.standardize_coords(sample["y_coord"]),
        )

    def get_target_frame(self, idx, frame_idx=0):
        target_path = self.targets_index[idx]["file"]
        target = torch.load(target_path)
        return target["y_frame"][frame_idx]

    def compute_coords_metrics(self):
        all_coords = []

        for idx in tqdm(range(len(self)), "Computing Metrics", leave=False):
            sample = torch.load(self.index[idx]["file"])
            all_coords.append(sample["X_coord"])
            all_coords.append(sample["y_coord"])

        all_coords = torch.cat(all_coords, dim=0).float()

        metrics = {
            "mean": all_coords.mean(dim=0),
            "std": all_coords.std(dim=0),
        }

        path = self.dataset_path / "metrics.pt"
        torch.save(metrics, path)
        return metrics

    def _load_or_compute_metrics(self):
        path = self.dataset_path / "metrics.pt"
        if path.exists():
            return torch.load(path)
        else:
            return self.compute_coords_metrics()

    def standardize_coords(self, coord):
        mean = self._metrics["mean"].unsqueeze(0)
        std = self._metrics["std"].unsqueeze(0)
        return (coord.float() - mean) / std

    def destandardize_coords(self, coord_std):
        mean = self._metrics["mean"].unsqueeze(0)
        std = self._metrics["std"].unsqueeze(0)
        return coord_std * std + mean

class AugmentedDataset(torch.utils.data.Dataset):

    def __init__(self, base_ds, coord_noise_std=0.02, temporal_keep=0.8, max_rotation_deg=5.0):
        self.base_ds = base_ds
        self.coord_noise_std = coord_noise_std
        self.temporal_keep = temporal_keep
        self.max_rotation_deg = max_rotation_deg
        self.frame_aug = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToDtype(torch.uint8, scale=True),
        ])

    def __len__(self):
        return len(self.base_ds)

    @staticmethod
    def _rotate_around_pivot(coords, pivot, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        centered = coords - pivot
        rotated = torch.stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a,
        ], dim=1)
        return rotated + pivot

    def __getitem__(self, idx):
        sample_id, X_frames, X_coords, y_coords = self.base_ds[idx]

        X_frames = self.frame_aug(X_frames)

        if random.random() < 0.5:
            T = X_frames.shape[0]
            n_keep = max(2, int(T * self.temporal_keep))
            indices = sorted(random.sample(range(T), n_keep))
            X_frames = X_frames[indices]
            X_coords = X_coords[indices]

        if random.random() < 0.5:
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            angle_rad = math.radians(angle)
            pivot = X_coords[-1:]
            X_coords = self._rotate_around_pivot(X_coords, pivot, angle_rad)
            y_coords = self._rotate_around_pivot(y_coords, pivot, angle_rad)

        X_coords = X_coords + torch.randn_like(X_coords) * self.coord_noise_std
        y_coords = y_coords + torch.randn_like(y_coords) * self.coord_noise_std

        if random.random() < 0.5:
            X_frames = X_frames.flip(-1)
            X_coords = X_coords.clone()
            X_coords[:, 0] = -X_coords[:, 0]
            y_coords = y_coords.clone()
            y_coords[:, 0] = -y_coords[:, 0]

        return (sample_id, X_frames, X_coords, y_coords)

def collate_fn(batch):
    X_frames, X_coords, y_coords = [], [], []
    X_l, y_l = [], []

    for obj in batch:
        X_frames.append(obj[1])
        X_coords.append(obj[2])
        y_coords.append(obj[3])
        X_l.append(obj[2].shape[0])
        y_l.append(obj[3].shape[0])

    X_frames = pad_sequence(X_frames, True, 0)
    X_coords = pad_sequence(X_coords, True, 0)
    y_coords = pad_sequence(y_coords, True, 0)
    X_l = torch.tensor(X_l)
    y_l = torch.tensor(y_l)

    return (
        X_frames, X_coords, y_coords,
        X_l, y_l
    )