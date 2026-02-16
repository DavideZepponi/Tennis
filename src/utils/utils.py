import torch

def predict_trajectory(model, device, ds, idx):

    model.eval()
    with torch.no_grad():
        _, X_frames, X_coords, y_coords = ds[idx]
        pred_coords, _ = model.predict(
            frames=X_frames.unsqueeze(0).to(device).float() / 255.0,
            device=device,
            encoder_trajectory=X_coords.unsqueeze(0).to(device),
            encoder_traj_lengths=torch.tensor([X_coords.shape[0]], device=device),
        )
        pred_coords = ds.destandardize_coords(pred_coords.squeeze(0).cpu()).int()
        gt_coords = ds.destandardize_coords(y_coords.cpu()).int()
        
        targets_index = torch.load(ds.dataset_path / "targets_frames" / "index.pt")
        target_sample = torch.load(targets_index[idx]["file"])
        bg_frame = target_sample["y_frame"][0]

    return bg_frame, gt_coords, pred_coords