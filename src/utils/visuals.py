import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import os

fig = None
axes = None

def plot_training(train_hist, valid_hist, ds, model, device, model_name):
    global fig, axes

    out_dir = os.path.join("artifacts", model_name)
    os.makedirs(out_dir, exist_ok=True)

    epoch = len(train_hist)

    if fig is None:
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].clear()
    axes[0].plot(train_hist, label="Train")
    axes[0].plot(valid_hist, label="Valid")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training Progress")

    axes[1].clear()

    model.eval()
    with torch.no_grad():
        _, X_frames, X_coords, y_coords = ds[0]

        pred_coords, _ = model.predict(
            frames=X_frames.unsqueeze(0).to(device).float() / 255.0,
            device=device,
            encoder_trajectory=X_coords.unsqueeze(0).to(device),
            encoder_traj_lengths=torch.tensor([X_coords.shape[0]], device=device),
        )

        pred_coords = ds.destandardize_coords(pred_coords.squeeze(0).cpu()).int()
        gt_coords = ds.destandardize_coords(y_coords.cpu()).int()

        targets_index = torch.load(ds.dataset_path / "targets_frames" / "index.pt")
        target_sample = torch.load(targets_index[0]["file"])
        bg_frame = target_sample["y_frame"][0]

    axes[1].imshow(bg_frame)
    axes[1].plot(gt_coords[:, 0], gt_coords[:, 1], label="Ground Truth", color="green", linewidth=2)
    axes[1].plot(pred_coords[:, 0], pred_coords[:, 1], label="Prediction", color="red", linewidth=2)
    axes[1].legend(loc="best")
    axes[1].set_title(f"Inference (Epoch {epoch})")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:04d}.png"), dpi=100)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)