from scipy.interpolate import interp1d
import torchvision.transforms.v2 as T
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import skimage
import torch
import yaml

def split_clip(clip: np.ndarray, lables: pd.DataFrame, adj_context: int):
    coords = lables[["x-coordinate", "y-coordinate"]].values
    split_idxs = np.where(lables["status"] == 1)[0]

    raw_frames = [s for s in np.split(clip, split_idxs) if s.size > 0]
    raw_coords = [s for s in np.split(coords, split_idxs) if s.size > 0]

    frames_context, frames_target = [], []
    coords_context, coords_target = [], []

    num_seqs = len(raw_frames)

    if num_seqs == 2:
        frames_context.append(raw_frames[0])
        frames_target.append(raw_frames[1])
        coords_context.append(raw_coords[0])
        coords_target.append(raw_coords[1])

    elif num_seqs >= 3:
        for i in range(1, num_seqs - 1):
            f_concat = np.concatenate([
                raw_frames[i-1][-adj_context:],
                raw_frames[i]
            ], axis=0)
            
            c_concat = np.concatenate([
                raw_coords[i-1][-adj_context:], 
                raw_coords[i]
            ], axis=0)
            
            frames_context.append(f_concat)
            frames_target.append(raw_frames[i+1])
            coords_context.append(c_concat)
            coords_target.append(raw_coords[i+1])

    return frames_context, frames_target, coords_context, coords_target

def load_clip(clip_path: Path, adj_context: int):
    frames_context_ds, frames_target_ds = {}, {}
    coords_context_ds, coords_target_ds = {}, {}

    iter_clip = sorted([d for d in clip_path.iterdir() if ".jpg" in d.name])

    clip = []

    for frame_path in iter_clip:
        clip.append(skimage.io.imread(frame_path))

    clip = np.array(clip)
    coords = pd.read_csv(clip_path.joinpath("Label.csv"))

    (
        frames_context, frames_target,
        coords_context, coords_target
    ) = split_clip(clip, coords, adj_context)
    
    for idx, (fc, ft, lc, lt) in enumerate(zip(
        frames_context, frames_target,
        coords_context, coords_target
    )):
        key = f"{clip_path.absolute()}.{idx}"
        frames_context_ds.update({key: fc})
        frames_target_ds.update({key: ft})
        coords_context_ds.update({key: lc})
        coords_target_ds.update({key: lt})

    dataset = {
        "context": {
            "frames": frames_context_ds,
            "coords": coords_context_ds
        },
        "target": {
            "frames": frames_target_ds,
            "coords": coords_target_ds
        }
    }

    return dataset

def preprocess_clip(dataset, max_consec_nan: int, min_seq_len: int, image_h: int, image_w: int):
    frame_transform = T.Compose([
        T.Resize((image_h, image_w))
    ])

    def interp(arr):
        indices = np.arange(len(arr))
        not_nan = ~np.isnan(arr)
        f = interp1d(indices[not_nan], arr[not_nan], kind='linear', fill_value="extrapolate")
        return f(indices)

    def keep(arr):
        if len(arr) < min_seq_len:
            return 0
        
        mask = np.isnan(arr)
        bounded = np.hstack(([False], mask, [False]))
        diffs = np.diff(bounded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        lengths = ends - starts

        return 0 if np.any(lengths > max_consec_nan) else 1

    ids = []
    X_frames, X_coords = [], []
    y_frames, y_coords = [], []

    for k, v in dataset["context"]["frames"].items():
        X_frame = v.copy()
        X_coord = dataset["context"]["coords"][k].copy()
        y_frame = dataset["target"]["frames"][k].copy()
        y_coord = dataset["target"]["coords"][k].copy()

        check_X = keep(X_coord[:, 0])
        check_y = keep(y_coord[:, 0])

        if check_X and check_y:
            X_coord[:, 0] = interp(X_coord[:, 0])
            X_coord[:, 1] = interp(X_coord[:, 1])
            y_coord[:, 0] = interp(y_coord[:, 0])
            y_coord[:, 1] = interp(y_coord[:, 1])

            X_frame = torch.tensor(X_frame, dtype=torch.uint8).permute(0, 3, 1, 2)  # (T, C, H, W)
            X_frame = frame_transform(X_frame)
            
            ids.append(k)
            X_frames.append(X_frame)
            X_coords.append(torch.tensor(X_coord, dtype=torch.int16))
            y_frames.append(torch.tensor(y_frame, dtype=torch.uint8))
            y_coords.append(torch.tensor(y_coord, dtype=torch.int16))

    return {
        "ids": ids,
        "X_frames": X_frames,
        "X_coords": X_coords,
        "y_frames": y_frames,
        "y_coords": y_coords
    }
    
def save_clip(
    dataset,
    path: Path,
    index_contexts: list,
    index_targets_frames: list
):
    path.mkdir(parents=True, exist_ok=True)
    contexts_dir = path / "context"
    contexts_dir.mkdir(exist_ok=True)
    targets_frames_dir = path / "targets_frames"
    targets_frames_dir.mkdir(exist_ok=True)

    for i in range(len(dataset["ids"])):
        context = {
            "id": dataset["ids"][i],
            "X_frame": dataset["X_frames"][i],
            "X_coord": dataset["X_coords"][i],
            "y_coord": dataset["y_coords"][i],
        }

        file_name = f"sample_{len(index_contexts):08d}.pt"
        file_path = contexts_dir / file_name
        
        torch.save(context, file_path)

        index_contexts.append({
            "id": dataset["ids"][i],
            "file": str(file_path)
        })

        target_frames = {
            "id": dataset["ids"][i],
            "y_frame": dataset["y_frames"][i]
        }

        file_name = f"sample_{len(index_targets_frames):08d}.pt"
        file_path = targets_frames_dir / file_name
        
        torch.save(target_frames, file_path)

        index_targets_frames.append({
            "id": dataset["ids"][i],
            "file": str(file_path)
        })

    return index_contexts, index_targets_frames

def preprocessing():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    path = Path(config["torch_dataset"]["path"]) / f"{config["torch_dataset"]["image_h"]}x{config["torch_dataset"]["image_w"]}"

    if Path(path).exists():
        print("\nDataset with same name found in folder. Skipping Dataset Creation.")
    else:
        index_contexts, index_targets_frames = [], []

        iter_dataset = sorted([d for d in Path(config["original_dataset"]["path"]).iterdir() if not d.is_file()])

        for game in tqdm(iter_dataset, desc="Processing"):
            iter_game = sorted([d for d in game.iterdir() if not d.is_file()])

            for clip_path in tqdm(iter_game, leave=False, desc="Game Processing"):

                dataset = load_clip(
                    clip_path = Path(clip_path),
                    adj_context=config["torch_dataset"]["adj_context"]
                )

                dataset = preprocess_clip(
                    dataset=dataset,
                    max_consec_nan=config["torch_dataset"]["max_consec_nan"],
                    min_seq_len=config["torch_dataset"]["min_seq_len"],
                    image_h=config["torch_dataset"]["image_h"],
                    image_w=config["torch_dataset"]["image_w"]
                )

                index_contexts, index_targets_frames = save_clip(
                    dataset=dataset,
                    path=Path(config["torch_dataset"]["path"]) / f"{config["torch_dataset"]["image_h"]}x{config["torch_dataset"]["image_w"]}",
                    index_contexts=index_contexts,
                    index_targets_frames=index_targets_frames
                )

        torch.save(index_contexts, path / "context/index.pt")
        torch.save(index_targets_frames, path / "targets_frames/index.pt")