"""
FastSpeech 2 Training Script for English (Svarah).

Implements:
1. Multi-task loss tracking (Total, Mel, Duration, Pitch, Energy).
2. TensorBoard logging for both Train and Validation curves.
3. Warmup learning rate schedule and gradient clipping.
4. Checkpoint saving and resumption via --restore_step.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import Dataset
from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from utils.tools import to_device, synth_one_sample
from utils.model import get_vocoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_configs(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["preprocess"], cfg["model"], cfg["train"]


def evaluate(model, step, configs, val_loader, val_dataset, loss_fn, vocoder, logger):
    """Runs evaluation over the validation split and logs to TensorBoard."""
    model.eval()
    loss_sums = [0.0] * 6
    total_samples = 0

    with torch.no_grad():
        for batchs in val_loader:
            for batch in batchs:
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = loss_fn(batch, output)
                b_size = len(batch[0])
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * b_size
                total_samples += b_size

    loss_means = [s / max(total_samples, 1) for s in loss_sums]
    (
        total_loss,
        mel_loss,
        postnet_loss,
        pitch_loss,
        energy_loss,
        duration_loss,
    ) = loss_means

    # Log Validation metrics to TensorBoard
    if logger is not None:
        logger.add_scalar("Validation/Total_Loss", total_loss, step)
        logger.add_scalar("Validation/Mel_Loss", mel_loss, step)
        logger.add_scalar("Validation/Mel_Postnet_Loss", postnet_loss, step)
        logger.add_scalar("Validation/Pitch_Loss", pitch_loss, step)
        logger.add_scalar("Validation/Energy_Loss", energy_loss, step)
        logger.add_scalar("Validation/Duration_Loss", duration_loss, step)

        # Synthesize sample visualization & audio if vocoder is available
        if vocoder is not None:
            try:
                preprocess_config, model_config, _ = configs
                fig, wav_recon, wav_pred, tag = synth_one_sample(
                    batch, output, vocoder, model_config, preprocess_config
                )
                logger.add_figure(f"Validation/Spec_{tag}", fig, step)
                sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                if wav_pred is not None:
                    logger.add_audio(f"Validation/Audio_{tag}", wav_pred / max(abs(wav_pred).max(), 1e-5), step, sample_rate=sr)
            except Exception as e:
                print(f"[Warning] Sample synthesis skipped during eval: {e}")

    model.train()
    return loss_means


def train(args):
    preprocess_config, model_config, train_config = load_configs(args.config)
    print(f"Device: {device}")
    print(f"Preprocessed Data: {preprocess_config['path']['preprocessed_path']}")
    print(f"Checkpoints Dir:  {train_config['path']['ckpt_path']}")

    # 1. Initialize Dataset & Dataloaders
    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # batches grouped by duration to minimize padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    # 2. Initialize Model, Optimizer, and Loss
    model = FastSpeech2(preprocess_config, model_config).to(device)
    scheduled_optim = ScheduledOptim(
        model, train_config, model_config, current_step=args.restore_step
    )
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # 3. Resume from Checkpoint if requested
    if args.restore_step > 0:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"], f"{args.restore_step}.pth.tar"
        )
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        scheduled_optim._optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Resumed from step {args.restore_step}")

    # 4. Optional Vocoder for Validation Audio Generation
    vocoder = None
    try:
        vocoder = get_vocoder(model_config, device)
        print("Pretrained vocoder loaded for validation synthesis.")
    except Exception as e:
        print(f"Vocoder not initialized (synthesis during eval disabled): {e}")

    # 5. TensorBoard Logger & Checkpoint Directory
    log_dir = train_config["path"]["log_path"]
    ckpt_dir = train_config["path"]["ckpt_path"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    # 6. Training Loop
    step = args.restore_step
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    val_step = train_config["step"]["val_step"]
    save_step = train_config["step"]["save_step"]
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]

    model.train()
    print("Beginning training loop...")
    epoch = 0

    while step < total_step:
        epoch += 1
        for batchs in train_loader:
            for batch in batchs:
                step += 1
                batch = to_device(batch, device)

                # Forward pass
                output = model(*(batch[2:]))
                losses = loss_fn(batch, output)
                total_loss = losses[0] / grad_acc_step

                # Backward pass
                total_loss.backward()

                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    scheduled_optim.step_and_update_lr()
                    scheduled_optim.zero_grad()

                # TensorBoard Training Log
                if step % log_step == 0:
                    lr = scheduled_optim._optimizer.param_groups[0]["lr"]
                    (
                        l_tot,
                        l_mel,
                        l_post,
                        l_pitch,
                        l_energy,
                        l_dur,
                    ) = [l.item() for l in losses]

                    logger.add_scalar("Train/Total_Loss", l_tot, step)
                    logger.add_scalar("Train/Mel_Loss", l_mel, step)
                    logger.add_scalar("Train/Mel_Postnet_Loss", l_post, step)
                    logger.add_scalar("Train/Pitch_Loss", l_pitch, step)
                    logger.add_scalar("Train/Energy_Loss", l_energy, step)
                    logger.add_scalar("Train/Duration_Loss", l_dur, step)
                    logger.add_scalar("Train/Learning_Rate", lr, step)

                    print(
                        f"Step {step}/{total_step} | "
                        f"Tot: {l_tot:.4f} | Mel: {l_mel:.4f} | Dur: {l_dur:.4f} | "
                        f"Pitch: {l_pitch:.4f} | Energy: {l_energy:.4f} | LR: {lr:.6f}"
                    )

                # Validation Loop
                if step % val_step == 0:
                    val_losses = evaluate(
                        model, step, (preprocess_config, model_config, train_config),
                        val_loader, val_dataset, loss_fn, vocoder, logger
                    )
                    print(
                        f"[Validation Step {step}] Tot: {val_losses[0]:.4f} | "
                        f"Mel: {val_losses[1]:.4f} | Dur: {val_losses[5]:.4f} | "
                        f"Pitch: {val_losses[3]:.4f} | Energy: {val_losses[4]:.4f}"
                    )

                # Checkpoint Saving
                if step % save_step == 0:
                    save_path = os.path.join(ckpt_dir, f"{step}.pth.tar")
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": scheduled_optim._optimizer.state_dict(),
                            "step": step,
                        },
                        save_path,
                    )
                    # Also write latest
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": scheduled_optim._optimizer.state_dict(),
                            "step": step,
                        },
                        os.path.join(ckpt_dir, "latest.pth.tar"),
                    )
                    print(f"Saved checkpoint: {save_path}")

                if step >= total_step:
                    break
            if step >= total_step:
                break

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastSpeech 2 on English (Svarah).")
    parser.add_argument("--config", default="English/config.yaml", help="Path to English/config.yaml")
    parser.add_argument("--restore_step", type=int, default=0, help="Checkpoint step to resume from")
    args = parser.parse_args()

    train(args)

