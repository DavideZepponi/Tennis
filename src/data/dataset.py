from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from pathlib import Path
import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path: Path):
        self.dataset_path = path
        self.index = torch.load(path / "context" / "index.pt")
        self.targets_index = torch.load(path / "targets_frames" / "index.pt")
        self._metrics = self._load_or_compute_metrics()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sample_path = self.index[idx]["file"]
        sample = torch.load(sample_path)

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

class FlippedDataset(torch.utils.data.Dataset):

    def __init__(self, base_ds):
        self.base_ds = base_ds
        ds = base_ds
        while isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        self.root_ds = ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample_id, X_frames, X_coords, y_coords = self.base_ds[idx]

        X_frames_flipped = X_frames.flip(-1)

        X_coords_flipped = X_coords.clone()
        X_coords_flipped[:, 0] = -X_coords_flipped[:, 0]

        y_coords_flipped = y_coords.clone()
        y_coords_flipped[:, 0] = -y_coords_flipped[:, 0]

        return (sample_id, X_frames_flipped, X_coords_flipped, y_coords_flipped)

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