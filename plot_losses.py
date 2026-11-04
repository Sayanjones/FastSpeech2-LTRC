"""
Loss Plotting Utility for FastSpeech 2.

Extracts real loss events from TensorBoard log files and generates:
  - Results/<Language>/training_loss.png
  - Results/<Language>/validation_loss.png

HARD RULE 2:
Never fabricate anything. Every loss value and plot must come from a real run.
If log files are not yet generated, informs the user rather than creating mock data.
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None


def extract_scalars_from_events(log_dir: str):
    """Parses TensorBoard event logs to extract scalar values."""
    if EventAccumulator is None:
        raise ImportError("Please install tensorboard: pip install tensorboard")

    event_files = glob.glob(os.path.join(log_dir, "**", "*events.out.tfevents*"), recursive=True)
    if not event_files:
        event_files = glob.glob(os.path.join(log_dir, "*events.out.tfevents*"))

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    event_files.sort(key=os.path.getmtime)
    print(f"Reading events from {len(event_files)} file(s) in {log_dir}...")

    # Load events
    ea = EventAccumulator(event_files[-1])
    ea.Reload()

    tags = ea.Tags()["scalars"]
    print(f"Found scalar tags: {tags}")

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = (steps, values)

    return data


def plot_losses(log_dir: str, output_dir: str, language: str = "English"):
    """Plots training_loss.png and validation_loss.png from real logs."""
    os.makedirs(output_dir, exist_ok=True)
    data = extract_scalars_from_events(log_dir)

    # 1. Plot Training Loss
    train_tags = [
        ("Train/Total_Loss", "Total Loss", "navy"),
        ("Train/Mel_Loss", "Mel Loss", "forestgreen"),
        ("Train/Duration_Loss", "Duration Loss", "darkorange"),
        ("Train/Pitch_Loss", "Pitch Loss", "crimson"),
        ("Train/Energy_Loss", "Energy Loss", "purple"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (tag, label, color) in enumerate(train_tags):
        ax = axes[idx]
        if tag in data:
            steps, vals = data[tag]
            ax.plot(steps, vals, label=label, color=color, linewidth=1.6)
            ax.set_title(f"{label} ({language})", fontweight="bold")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            ax.grid(linestyle="--", alpha=0.5)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"Tag {tag} not in logs", ha="center", va="center")

    # Overall Training Overview
    ax_main = axes[5]
    if "Train/Total_Loss" in data:
        steps, vals = data["Train/Total_Loss"]
        ax_main.plot(steps, vals, color="navy", label="Train Total")
    if "Validation/Total_Loss" in data:
        v_steps, v_vals = data["Validation/Total_Loss"]
        ax_main.plot(v_steps, v_vals, color="red", linestyle="--", label="Val Total")
    ax_main.set_title("Train vs Validation Convergence", fontweight="bold")
    ax_main.set_xlabel("Steps")
    ax_main.set_ylabel("Total Loss")
    ax_main.grid(linestyle="--", alpha=0.5)
    ax_main.legend()

    plt.tight_layout()
    train_out = os.path.join(output_dir, "training_loss.png")
    plt.savefig(train_out, dpi=200)
    plt.close()
    print(f"Generated Training Loss Plot: {train_out}")

    # 2. Plot Validation Loss
    val_tags = [
        ("Validation/Total_Loss", "Val Total Loss", "navy"),
        ("Validation/Mel_Loss", "Val Mel Loss", "forestgreen"),
        ("Validation/Duration_Loss", "Val Duration Loss", "darkorange"),
        ("Validation/Pitch_Loss", "Val Pitch Loss", "crimson"),
        ("Validation/Energy_Loss", "Val Energy Loss", "purple"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (tag, label, color) in enumerate(val_tags):
        ax = axes[idx]
        if tag in data:
            steps, vals = data[tag]
            ax.plot(steps, vals, label=label, color=color, linewidth=1.6, marker="o", markersize=3)
            ax.set_title(f"{label} ({language})", fontweight="bold")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            ax.grid(linestyle="--", alpha=0.5)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"Tag {tag} not in logs", ha="center", va="center")

    axes[5].axis("off")
    plt.tight_layout()
    val_out = os.path.join(output_dir, "validation_loss.png")
    plt.savefig(val_out, dpi=200)
    plt.close()
    print(f"Generated Validation Loss Plot: {val_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation losses from real TensorBoard logs.")
    parser.add_argument("--log_dir", default=None, help="Path to log directory")
    parser.add_argument("--output_dir", default=None, help="Path to output directory")
    parser.add_argument("--language", "--lang", default="English", help="Language (English or Telugu)")
    args = parser.parse_args()

    log_dir = args.log_dir if args.log_dir else f"{args.language}/log"
    output_dir = args.output_dir if args.output_dir else f"Results/{args.language}"

    plot_losses(log_dir, output_dir, args.language)

