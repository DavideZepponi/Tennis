# [Dataset](https://www.kaggle.com/datasets/sofuskonglevoll/tracknet-tennis?resource=download)

# Model

A transformer that predicts where a tennis ball will go after a shot is hit. The model watches video frames leading up to the shot considers the ball's trajectory before the shot, and autoregressively generates the predicted post-shot trajectory as a sequence of (x, y) coordinates.

## Architecture

### Encoder

#### Video Frames

Raw video frames are passed through a **frozen pretrained ResNet-18**. This CNN extracts a 512-dimensional feature vector per frame, which is then linearly projected down to `d_model` dimensions. ResNet-18 is frozen by default so the model trains faster and avoids overfitting on small datasets. It can be unfreeze for fine-tuning, or it can be swapped to a stronger CNN (e.g., ResNet-50, ViT, TimeSformer).

#### Pre-Shot Ball Trajectory

If available, the ball's (x, y) positions before the shot are projected to `d_model` via a linear layer. These tokens are concatenated after the visual tokens to form a single sequence.

#### Encoding

Each token receives two additive embeddings before entering the Transformer:

* Sinusoidal positional encoding.
* Modality embedding: a learned embedding that tells the model *what kind* of token this is (video vs. trajectory). Without this, the Transformer wouldn't know where the visual tokens end and trajectory tokens begin.

The combined token sequence passes through a standard Transformer encoder (self-attention + feedforward, repeated `n_layers` times). The output is the **memory** tensor that the decoder will attend to.

### Decoder

#### Inputs and Outputs

During training, the decoder receives the ground-truth post-shot trajectory shifted right by one position (teacher forcing). At each position, it predicts the next (x, y) coordinate.

During inference, the decoder starts with the ball's last known position and feeds its own predictions back as input, one step at a time.

#### Dual Output Heads

The decoder has two output heads:

1. **Coordinate head** : `d_model → d_model → 2`. Predicts (x, y) at each timestep.
2. **Stop head** : `d_model → d_model/2 → 1`. Predicts the probability that the trajectory ends at this step. This replaces a discrete "stop token".

## Training

### Loss Function

The total loss is a weighted sum of two components:

```
loss = coord_loss + λ_stop × stop_loss
```

* **`coord_loss`** : Smooth L1 loss between predicted and ground-truth (x, y) coordinates.
* **`stop_loss`** : Binary cross-entropy with logits. Labels are 0 at every real step except the last, which is 1.
* **`λ_stop`** : Weighting factor (default 0.5).

### Optimizer and Schedule

* AdamW
* Warmup + cosine annealing.
* Gradient clipping.
