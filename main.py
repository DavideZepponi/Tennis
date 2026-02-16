from src.model.transformer import TrajectoryTransformer
from src.data.preprocessing import preprocessing
from src.data.dataset import Dataset
from src.training import train
from pathlib import Path
import yaml

def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    preprocessing()

    dataset = Dataset(Path(config["torch_dataset"]["path"]) / f"{config["torch_dataset"]["image_h"]}x{config["torch_dataset"]["image_w"]}")

    models = {
        "convnext_tiny_freeze_v00": TrajectoryTransformer(
            d_model=256,
            nhead_encoder=8,
            nhead_decoder=8,
            n_encoder_layers=4,
            n_decoder_layers=6,
            dropout=0.1,
            freeze_backbone=True,
        )
    }

    train(
        dataset,
        models,
        batch_size=1,
        warmup_epochs=10,
        num_epochs=300,
        patience=100
    )

if __name__ == "__main__":
    main()