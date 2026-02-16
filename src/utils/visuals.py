from src.utils.utils import predict_trajectory
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

BG_WINDOW = '#FFFFFF'     
BG_CARD = '#EBEBEB'
ACCENT_BLUE = '#99C24D'    
ACCENT_ORANGE = '#E3170A'  
ACCENT_GREEN = '#99C24D'
ACCENT_PINK = '#E3170A'    
TEXT_MAIN = '#3A3A3C'
TEXT_SECONDARY = '#8E8E93'

mpl.rcParams['font.family'] = 'monospace'
mpl.rcParams['font.monospace'] = ['Consolas', 'Courier New', 'DejaVu Sans Mono', 'Monaco']

fig = None
axes = None
footer_text = None 

def plot_training(train_hist, valid_hist, ds, model, device, model_name):
    global fig, axes, footer_text

    out_dir = os.path.join("artifacts", model_name)
    os.makedirs(out_dir, exist_ok=True)
    epoch_count = len(train_hist)

    bg_frame, gt_coords, pred_coords = predict_trajectory(model, device, ds, 266)

    img_h, img_w = bg_frame.shape[:2]
    aspect_ratio = img_w / img_h 

    if fig is None:
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor=BG_WINDOW)
        footer_text = fig.text(0.5, 0.05, "", ha='center', fontsize=14, 
                               color=TEXT_SECONDARY, fontweight='500')
    
    for ax in axes:
        ax.clear()
        ax.set_box_aspect(1/aspect_ratio) 
        ax.set_facecolor(BG_CARD)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- LEFT: Training Metrics ---
    ax0 = axes[0]
    ax0.grid(True, axis='y', color='#3A3A3C', linestyle='--', linewidth=0.2, alpha=0.5, zorder=0)
    
    epochs = np.arange(epoch_count)
    ax0.plot(epochs, train_hist, color=ACCENT_BLUE, linewidth=3, label="Training", zorder=3)
    ax0.plot(epochs, valid_hist, color=ACCENT_ORANGE, linewidth=3, label="Validation", zorder=4)

    # Fix Scatter: Now always on the last added metrics
    if epoch_count > 0:
        last_epoch = epochs[-1]
        ax0.scatter(last_epoch, train_hist[-1], color='#3A3A3C', s=40, zorder=5, edgecolors=BG_CARD)
        ax0.scatter(last_epoch, valid_hist[-1], color='#3A3A3C', s=40, zorder=5, edgecolors=BG_CARD)

    # Fix H-Axis: Force integer ticks
    ax0.set_xticks(epochs)
    ax0.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    ax0.set_title("Progress", loc='left', fontsize=20, pad=25)
    ax0.text(0, 1.02, "Loss x Epoch", transform=ax0.transAxes, 
         fontsize=12, verticalalignment='bottom')
    
    # Legend below the graph
    ax0.legend(loc='upper center', 
                bbox_to_anchor=(0.5, -0.15), # Standardized offset
                ncol=2, 
                frameon=True,
                fontsize=11,
                handletextpad=0.5,
                columnspacing=1.5)

    # --- RIGHT: Inference ---
    ax1 = axes[1]
    ax1.imshow(bg_frame)
    
    ax1.plot(gt_coords[:, 0], gt_coords[:, 1], color=ACCENT_GREEN, 
             linewidth=3, label="Ground Truth", solid_capstyle='round')
    ax1.plot(pred_coords[:, 0], pred_coords[:, 1], color=ACCENT_PINK, 
             linewidth=2, label="Prediction", linestyle='-', solid_capstyle='round')

    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Inference", loc='left', fontsize=20, pad=25)
    ax1.text(0, 1.02, f"Epoch {epoch_count - 1}", transform=ax1.transAxes, 
         fontsize=12, verticalalignment='bottom')
    
    # Legend below the image box
    ax1.legend(loc='upper center', 
                bbox_to_anchor=(0.5, -0.15), # Matches ax0 exactly
                ncol=2, 
                frameon=True, 
                fontsize=11,
                handletextpad=0.5,
                columnspacing=1.5)

    # --- UPDATE FOOTER ---
    if valid_hist:
        footer_text.set_text(f"Current Val. Loss: {valid_hist[-1]:.4f}   |   Best Val. Loss: {min(valid_hist):.4f}   |   Best Train. Loss: {min(train_hist):.4f}")

    # Adjusted bottom margin to accommodate legends below axes
    plt.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.25, wspace=0.2)
    
    fig.canvas.draw()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch_count - 1:04d}.png"), dpi=120, facecolor=BG_WINDOW)
    fig.canvas.flush_events()
    plt.pause(0.01)