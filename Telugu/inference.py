"""
Inference Script for FastSpeech 2 (Telugu).

Synthesizes speech for all 5 Telugu test sentences from the assignment specification:
  - T1: న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు.
  - T2a: తాడుపై అతని పట్టుచాలా బలంగా ఉంది.
  - T2b: ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది.
  - T3: పులి అడ విలో నిశ్శబ్ంద గా నడిచింది.
  - T4: ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి.

Supports:
  - --mode verbatim (default per Hard Rule 4)
  - --mode repaired (for comparative error analysis in Technical Report)

Outputs:
  - Audio: Results/Telugu/test_generated_audios/{id}.wav
  - Plots: Results/Telugu/variance_adaptor_plots/{duration,pitch,energy}/{id}_*.png
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from text import text_to_sequence
from text.telugu import repair_scrambled_telugu
from model import FastSpeech2
from utils.model import get_vocoder, vocoder_infer
from utils.tools import to_device
from test_sentences import TELUGU_TEST_SENTENCES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_configs(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["preprocess"], cfg["model"], cfg["train"]


def load_model(checkpoint_path: str, configs):
    preprocess_config, model_config, _ = configs
    model = FastSpeech2(preprocess_config, model_config).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading Telugu weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    else:
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            "Please train the model first or provide a valid checkpoint with --checkpoint."
        )
    model.eval()
    return model


def plot_duration(tokens, durations, out_path, title):
    """Plots predicted grapheme duration bar chart."""
    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.25), 4))
    x = np.arange(len(tokens))
    ax.bar(x, durations, color="teal", edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=75, fontsize=8)
    ax.set_ylabel("Duration (Mel Frames)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pitch(pitch_contour, out_path, title):
    """Plots predicted fundamental frequency (F0) contour."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    frames = np.arange(len(pitch_contour))
    ax.plot(frames, pitch_contour, color="darkviolet", linewidth=1.8, label="Predicted F0")
    ax.set_xlabel("Time (Mel Frames)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_energy(energy_contour, out_path, title):
    """Plots predicted frame-level acoustic energy contour."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    frames = np.arange(len(energy_contour))
    ax.plot(frames, energy_contour, color="forestgreen", linewidth=1.8, label="Predicted Energy")
    ax.set_xlabel("Time (Mel Frames)")
    ax.set_ylabel("Acoustic Energy")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def synthesize(args):
    configs = load_configs(args.config)
    preprocess_config, model_config, train_config = configs

    audio_dir = os.path.join(args.output_dir, "test_generated_audios")
    dur_dir = os.path.join(args.output_dir, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(args.output_dir, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(args.output_dir, "variance_adaptor_plots", "energy")
    for d in [audio_dir, dur_dir, pitch_dir, energy_dir]:
        os.makedirs(d, exist_ok=True)

    model = load_model(args.checkpoint, configs)
    vocoder = get_vocoder(model_config, device)

    cleaner_name = "telugu_cleaners" if args.mode == "verbatim" else "telugu_cleaners_with_repair"
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    stats_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)

    print(f"\n--- Running Telugu FastSpeech 2 Inference [Mode: {args.mode}] ---")

    for sid, raw_text in TELUGU_TEST_SENTENCES.items():
        text = raw_text if args.mode == "verbatim" else repair_scrambled_telugu(raw_text)
        print(f"\n[{sid}] Input: {text}")

        phone_ids = np.array(text_to_sequence(text, [cleaner_name]))
        tokens = [f"char_{i}" for i in range(len(phone_ids))]

        text_tensor = torch.from_numpy(phone_ids).long().unsqueeze(0).to(device)
        src_len = torch.tensor([len(phone_ids)]).long().to(device)
        speaker_id = torch.tensor([0]).long().to(device)

        max_src_len = torch.max(src_len).item()
        with torch.no_grad():
            output = model(
                speaker_id,
                text_tensor,
                src_len,
                max_src_len,
                p_control=args.pitch_control,
                e_control=args.energy_control,
                d_control=args.duration_control,
            )

            (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                _,
                src_masks,
                mel_masks,
                _,
                mel_lens,
            ) = output

            # 1. Vocode audio
            mel_len = mel_lens[0].item()
            mel = postnet_mel_predictions[0, :mel_len].unsqueeze(0).transpose(1, 2)
            wav = vocoder_infer(mel, vocoder, model_config, preprocess_config)[0]

            suffix = "" if args.mode == "verbatim" else "_repaired"
            wav_path = os.path.join(audio_dir, f"{sid}{suffix}.wav")
            sf.write(wav_path, wav, sr)
            print(f"  -> Generated Audio: {wav_path} ({len(wav) / sr:.2f}s)")

            # 2. Extract and Plot Duration
            pred_dur = torch.clamp(
                torch.round(torch.exp(log_duration_predictions[0, :src_len[0]]) - 1), min=0
            ).cpu().numpy().astype(int)

            dur_path = os.path.join(dur_dir, f"{sid}{suffix}_duration.png")
            plot_duration(tokens, pred_dur, dur_path, f"Predicted Telugu Grapheme Durations [{sid}]")
            print(f"  -> Saved Duration Plot: {dur_path}")

            # 3. Extract and Plot Pitch (F0)
            pred_pitch = pitch_predictions[0, :src_len[0]].cpu().numpy()
            if stats is not None and preprocess_config["preprocessing"]["pitch"]["normalization"]:
                pitch_mean = stats["pitch"][2]
                pitch_std = stats["pitch"][3]
                pred_pitch = pred_pitch * pitch_std + pitch_mean

            pitch_path = os.path.join(pitch_dir, f"{sid}{suffix}_pitch.png")
            plot_pitch(pred_pitch, pitch_path, f"Predicted Telugu Pitch (F0) Contour [{sid}]")
            print(f"  -> Saved Pitch Plot:    {pitch_path}")

            # 4. Extract and Plot Energy
            pred_energy = energy_predictions[0, :src_len[0]].cpu().numpy()
            if stats is not None and preprocess_config["preprocessing"]["energy"]["normalization"]:
                energy_mean = stats["energy"][2]
                energy_std = stats["energy"][3]
                pred_energy = pred_energy * energy_std + energy_mean

            energy_path = os.path.join(energy_dir, f"{sid}{suffix}_energy.png")
            plot_energy(pred_energy, energy_path, f"Predicted Telugu Acoustic Energy [{sid}]")
            print(f"  -> Saved Energy Plot:   {energy_path}")

    print("\nTelugu synthesis completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastSpeech 2 Inference on Telugu Test Sentences.")
    parser.add_argument("--config", default="Telugu/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="Telugu/checkpoints/latest.pth.tar", help="Path to checkpoint")
    parser.add_argument("--output_dir", default="Results/Telugu", help="Output directory")
    parser.add_argument("--mode", choices=["verbatim", "repaired"], default="verbatim", help="Synthesis mode")
    parser.add_argument("--pitch_control", type=float, default=1.0, help="Pitch control factor")
    parser.add_argument("--energy_control", type=float, default=1.0, help="Energy control factor")
    parser.add_argument("--duration_control", type=float, default=1.0, help="Speed/duration factor")
    args = parser.parse_args()

    synthesize(args)

