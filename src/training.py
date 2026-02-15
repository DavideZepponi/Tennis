from src.data.dataset import collate_fn, FlippedDataset
from torch.utils.data import DataLoader, random_split
from src.utils.visuals import plot_training
from torch.utils.data import ConcatDataset
import torch.optim as optim
from tqdm.auto import tqdm
from torch import nn
import random
import torch
import os

class EfficientSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Get lengths by peeking at actual items — works with any dataset type
        self.lengths = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            n_frames = item[1].shape[0]
            n_target = item[3].shape[0]
            self.lengths.append(n_frames + n_target)

        self.sorted_indices = sorted(
            range(len(self.lengths)),
            key=lambda i: self.lengths[i],
        )
        self.buckets = [
            self.sorted_indices[i : i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]

    def __iter__(self):
        buckets = [b.copy() for b in self.buckets]
        if self.shuffle:
            random.shuffle(buckets)
        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)
            yield from bucket

    def __len__(self):
        return len(self.dataset)

def train_step(model, batch, optimizer, loss_fn, device):
    model.train()

    (
        X_frames, X_coords, y_coords,
        X_l, y_l
    ) = batch

    X_frames = X_frames.to(device).float()
    X_coords = X_coords.to(device)
    y_coords = y_coords.to(device)
    X_l = X_l.to(device)
    y_l = y_l.to(device)

    y_coords_input = y_coords[:, :-1]
    y_coords_output = y_coords[:, 1:]

    dec_lengths = y_l - 1

    coords, stop_logits = model.forward(
        frames=X_frames,
        tgt=y_coords_input,
        device=device,
        frame_lengths=X_l,
        tgt_lengths=dec_lengths,
        encoder_trajectory=X_coords,
        encoder_traj_lengths=X_l
    )

    B, T, _ = coords.shape

    valid = torch.arange(T, device=device).unsqueeze(0) < dec_lengths.unsqueeze(1)

    coord_loss = loss_fn(coords[valid], y_coords_output[valid])

    stop_labels = torch.zeros(B, T, 1, device=device)
    stop_labels[torch.arange(B, device=device), dec_lengths - 1, 0] = 1.0

    stop_loss = nn.functional.binary_cross_entropy_with_logits(
        stop_logits[valid], stop_labels[valid],
    )

    lambda_stop = 0.5
    loss = coord_loss + lambda_stop * stop_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

def evaluate(model, valid_dl, loss_fn, device):
    model.eval()

    with torch.no_grad():
        total_coord_loss, total_stop_loss = 0.0, 0.0
        total_steps = 0

        for batch in tqdm(valid_dl, desc="Validation", leave=False):
            (
                X_frames, X_coords, y_coords,
                X_l, y_l
            ) = batch

            X_frames = X_frames.to(device).float() / 255.0
            X_coords = X_coords.to(device)
            y_coords = y_coords.to(device)
            X_l = X_l.to(device)
            y_l = y_l.to(device)

            y_coords_input = y_coords[:, :-1]
            y_coords_output = y_coords[:, 1:]
            dec_lengths = y_l - 1

            coords, stop_logits = model.forward(
                frames=X_frames,
                tgt=y_coords_input,
                device=device,
                frame_lengths=X_l,
                tgt_lengths=dec_lengths,
                encoder_trajectory=X_coords,
                encoder_traj_lengths=X_l
            )

            B, T, _ = coords.shape
            valid = torch.arange(T, device=device).unsqueeze(0) < dec_lengths.unsqueeze(1)
            n_valid = valid.sum().item()

            coord_loss = loss_fn(coords[valid], y_coords_output[valid])
            
            stop_labels = torch.zeros(B, T, 1, device=device)
            stop_labels[torch.arange(B, device=device), dec_lengths - 1, 0] = 1.0

            stop_loss = nn.functional.binary_cross_entropy_with_logits(
                stop_logits[valid], stop_labels[valid],
            )

            total_coord_loss += coord_loss.item() * n_valid
            total_stop_loss += stop_loss.item() * n_valid
            total_steps += n_valid

        avg_coord_loss = total_coord_loss / total_steps
        avg_stop_loss = total_stop_loss / total_steps
        avg_total_loss = avg_coord_loss + 0.5 * avg_stop_loss

    return avg_total_loss

def training_routine(name, model, loss_fn, optimizer, device, num_epochs, train_dl, valid_dl, ds, patience=5, min_delta=0.0, scheduler=None):
    loss_train_hist, loss_valid_hist = [], []
    best_model_state, best_loss, no_improve = None, float("inf"), 0

    for epoch in tqdm(range(num_epochs), desc="Overall Progress", unit="epoch"):

        # --- Training ---
        total_loss, total_steps = 0, 0

        for batch in tqdm(train_dl, desc="Training", leave=False):
            loss = train_step(
                model, batch, optimizer, loss_fn, device,
            )
            batch_size = batch[1].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            total_loss += loss * batch_size
            total_steps += batch_size

        avg_loss_train = total_loss / total_steps
        loss_train_hist.append(avg_loss_train)

        # --- Validation ---
        avg_loss_valid = evaluate(
            model, valid_dl, loss_fn, device,
        )
        loss_valid_hist.append(avg_loss_valid)

        plot_training(loss_train_hist, loss_valid_hist, ds, model, device, name)

        if scheduler is not None:
            scheduler.step()

        # --- Early Stopping ---
        improved = avg_loss_valid < (best_loss - min_delta)
        if improved:
            best_loss = avg_loss_valid
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        # --- Logging ---
        message = "☆" if improved else ""
        tqdm.write(
            f"EPOCH {epoch+1} | "
            f"TRAIN total: {avg_loss_train:.3e} | "
            f"VALID total: {avg_loss_valid:.3e} "
            f"{message}"
        )

        if no_improve >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1}")
            break

    os.makedirs("checkpoints", exist_ok=True)
    filename = os.path.join("checkpoints", f"{name}.pt")
    torch.save({
        "model_state": best_model_state,
        "num_epochs": epoch + 1,
        "loss_train_hist": loss_train_hist,
        "loss_valid_hist": loss_valid_hist
    }, filename)

def train(
    dataset,
    models,
    batch_size,
    num_epochs,
    patience
):
    torch.manual_seed(0)

    batch_size = batch_size
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flipped_ds = FlippedDataset(dataset)
    dataset_daug = ConcatDataset([dataset, flipped_ds])

    train_ds, valid_ds = random_split(dataset_daug, [0.9, 0.1])

    sampler = EfficientSampler(train_ds, batch_size=batch_size, shuffle=True)

    num_workers = max(1, os.cpu_count() - 2)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
    )

    for name, model in models.items():
        model = model.to(device)
        num_epochs = num_epochs

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        loss_fn = torch.nn.SmoothL1Loss()

        warmup_epochs = 5
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs])

        training_routine(
            name, model, loss_fn, optimizer, device, num_epochs,
            train_dl, valid_dl, dataset, patience, scheduler=scheduler,
        )