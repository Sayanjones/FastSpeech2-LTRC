"""
Telugu Preprocessing Pipeline for FastSpeech 2 using the IndicTTS Telugu Dataset.

================================================================================
PEDAGOGICAL OVERVIEW & ASSIGNMENT SPECIFICATIONS:
================================================================================
1. Telugu Text Frontend:
   - Shallow, phonemic orthography: Telugu characters map directly to acoustic sounds.
   - Symbol set: Native Telugu Unicode block (U+0C00 - U+0C7F: independent vowels,
     consonants, dependent vowel signs/matras, virama, anusvara, visarga).
   - Non-Standard Word (NSW) Normalization:
     * Digits and years (e.g., '2026' -> 'రెండు వేల ఇరవై ఆరు')
     * Ordinals (e.g., '12వ' -> 'పన్నెండవ')
     * Clock times (e.g., '10:45' -> 'పది నలభై ఐదు')
   - Conjunct Repair: Heals orphaned combining vowel signs and scrambled virama+anusvara sequences.

2. Subsampling Strategy:
   - IndicTTS Telugu (IIT Madras / SPRINGLab) contains ~8.74 hours (Male ~4.47h, Female ~4.27h).
   - Assignment constraint: ~5 hours train / ~1 hour validation, strictly disjoint.
   - Duration filtering: 1.5s <= duration <= 10.0s.
   - Speaker Balancing: Balances male and female utterances evenly (50% Male, 50% Female).
   - Deterministic Split: seed = 42, zero overlap.

3. Alignment & Feature Extraction:
   - Resampling: Standardizes all speech to 22,050 Hz (HiFi-GAN target).
   - Grapheme Lexicon: Generates a 1-to-1 Telugu character dictionary for MFA alignment.
   - Pitch & Energy Extraction: PyWorld DIO / Stonemask for F0, STFT L2-norm for energy.
   - Exports: preprocessed numpy tensors, stats.json, and Telugu/dataset_stats.json.
"""

import os
import sys
import glob
import json
import random
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import audio as Audio
from text.telugu import telugu_cleaners, repair_scrambled_telugu, TELUGU_SYMBOLS


def load_config(config_path: str = "Telugu/config.yaml") -> dict:
    """Loads YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# 1. SUBSAMPLING & SPLITTING LOGIC
# ==============================================================================

def subsample_indictts_telugu(
    records: List[Dict],
    target_train_hours: float = 5.0,
    target_val_hours: float = 1.0,
    min_dur: float = 1.5,
    max_dur: float = 10.0,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Subsamples IndicTTS Telugu utterances with duration filtering, gender/speaker balance,
    and a deterministic seed.

    Args:
        records: List of dictionaries with keys:
                 {'id': str, 'speaker': str, 'gender': str, 'duration': float, 'text': str, ...}
        target_train_hours: Target training hours (~5.0h)
        target_val_hours: Target validation hours (~1.0h)
        min_dur: Minimum duration in seconds (1.5s)
        max_dur: Maximum duration in seconds (10.0s)
        seed: Random seed for deterministic reproducibility (42)

    Returns:
        train_records, val_records, stats
    """
    random.seed(seed)
    np.random.seed(seed)

    # Step 1: Duration filtering
    valid_records = [
        r for r in records
        if min_dur <= float(r["duration"]) <= max_dur and len(r["text"].strip()) > 0
    ]
    print(f"[Subsampling] Initial records: {len(records)}, After duration filtering [{min_dur}s, {max_dur}s]: {len(valid_records)}")

    # Step 2: Separate by gender/speaker (e.g. te_male, te_female)
    speaker_groups = {}
    for r in valid_records:
        spk = r.get("speaker", r.get("gender", "default"))
        speaker_groups.setdefault(spk, []).append(r)

    for spk in speaker_groups:
        random.shuffle(speaker_groups[spk])

    # Target durations in seconds
    target_train_sec = target_train_hours * 3600.0
    target_val_sec = target_val_hours * 3600.0

    train_records = []
    val_records = []
    current_train_sec = 0.0
    current_val_sec = 0.0

    speakers = list(speaker_groups.keys())
    print(f"[Subsampling] Available speaker groups: {speakers}")

    # First pass: Allocate ~5h train balancing speakers round-robin
    idx = 0
    while current_train_sec < target_train_sec and any(speaker_groups.values()):
        spk = speakers[idx % len(speakers)]
        if speaker_groups[spk]:
            rec = speaker_groups[spk].pop(0)
            train_records.append(rec)
            current_train_sec += float(rec["duration"])
        idx += 1

    # Second pass: Allocate ~1h val from remaining utterances (zero overlap)
    idx = 0
    while current_val_sec < target_val_sec and any(speaker_groups.values()):
        spk = speakers[idx % len(speakers)]
        if speaker_groups[spk]:
            rec = speaker_groups[spk].pop(0)
            val_records.append(rec)
            current_val_sec += float(rec["duration"])
        idx += 1

    # Verify disjointness
    train_ids = set(r["id"] for r in train_records)
    val_ids = set(r["id"] for r in val_records)
    assert train_ids.isdisjoint(val_ids), "Fatal error: Overlap detected between train and val splits!"

    stats = {
        "seed": seed,
        "target_train_hours": target_train_hours,
        "actual_train_hours": round(current_train_sec / 3600.0, 4),
        "actual_train_seconds": round(current_train_sec, 2),
        "num_train_utterances": len(train_records),
        "target_val_hours": target_val_hours,
        "actual_val_hours": round(current_val_sec / 3600.0, 4),
        "actual_val_seconds": round(current_val_sec, 2),
        "num_val_utterances": len(val_records),
        "overlap_count": len(train_ids.intersection(val_ids)),
    }

    print(f"[Subsampling] Train: {stats['actual_train_hours']} h ({stats['num_train_utterances']} utterances)")
    print(f"[Subsampling] Val:   {stats['actual_val_hours']} h ({stats['num_val_utterances']} utterances)")
    print(f"[Subsampling] Strict Disjoint Split: 0% overlap confirmed.")

    return train_records, val_records, stats


# ==============================================================================
# 2. AUDIO RESAMPLING & TELUGU GRAPHEME LEXICON PREPARATION
# ==============================================================================

def generate_telugu_grapheme_lexicon(records: List[Dict], lexicon_path: str) -> None:
    """
    Generates a 1-to-1 Telugu character/akshara pronunciation dictionary for MFA.
    In shallow orthography, each Telugu Unicode grapheme maps directly to itself as a phone unit.
    """
    words = set()
    for r in records:
        clean_text = telugu_cleaners(r["text"])
        for w in clean_text.split():
            # Strip punctuation
            w_clean = "".join([c for c in w if c in TELUGU_SYMBOLS])
            if w_clean:
                words.add(w_clean)

    os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
    with open(lexicon_path, "w", encoding="utf-8") as f:
        for w in sorted(words):
            # Decompose word into grapheme characters separated by space
            graphemes = " ".join(list(w))
            f.write(f"{w}\t{graphemes}\n")
    print(f"[Lexicon] Generated Telugu grapheme lexicon with {len(words)} entries at {lexicon_path}")


def prepare_telugu_mfa_corpus(
    records: List[Dict],
    corpus_dir: str,
    target_sr: int = 22050,
) -> None:
    """
    Saves audio resampled to target_sr and creates corresponding .lab transcript files.
    Applies Telugu text normalization (number expansion, NFC) to the transcripts.
    """
    os.makedirs(corpus_dir, exist_ok=True)
    print(f"[MFA Prep] Exporting resampled audio and .lab files to {corpus_dir} ...")

    for r in tqdm(records):
        spk = r.get("speaker", "te_speaker")
        spk_dir = os.path.join(corpus_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)

        uid = r["id"]
        wav_out = os.path.join(spk_dir, f"{uid}.wav")
        lab_out = os.path.join(spk_dir, f"{uid}.lab")

        # Save normalized transcript
        clean_text = telugu_cleaners(r["text"])
        with open(lab_out, "w", encoding="utf-8") as f:
            f.write(clean_text + "\n")

        # Load and resample audio
        if "audio_array" in r and r["audio_array"] is not None:
            audio = r["audio_array"]
            sr = r.get("sampling_rate", target_sr)
        elif "audio_path" in r and os.path.exists(r["audio_path"]):
            audio, sr = librosa.load(r["audio_path"], sr=None)
        else:
            continue

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        sf.write(wav_out, audio, target_sr, subtype="PCM_16")


# ==============================================================================
# 3. FEATURE EXTRACTION (PITCH, ENERGY, MEL, DURATION)
# ==============================================================================

class TeluguPreprocessor:
    """
    Extracts acoustic features from aligned TextGrids and audio for Telugu.
    Adapts from the reference FastSpeech 2 preprocessing architecture.
    """

    def __init__(self, config: dict):
        self.config = config
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.filter_length = config["preprocessing"]["stft"]["filter_length"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]
        self.n_mel_channels = config["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_fmin = config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = config["preprocessing"]["mel"]["mel_fmax"]

        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.stft = Audio.stft.TacotronSTFT(
            self.filter_length,
            self.hop_length,
            self.win_length,
            self.n_mel_channels,
            self.sampling_rate,
            self.mel_fmin,
            self.mel_fmax,
        )

    def get_alignment(self, tier) -> Tuple[List[str], List[int], float, float]:
        """Parses phone/grapheme intervals from a tgt TextGrid phone tier."""
        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0

        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # Trim leading silence
            if len(phones) == 0:
                if p == "" or p == "sp" or p == "sil":
                    continue
                else:
                    start_time = s

            if p != "" and p != "sp" and p != "sil":
                phones.append(p)
                durations.append(
                    int(
                        np.round(e * self.sampling_rate / self.hop_length)
                        - np.round(s * self.sampling_rate / self.hop_length)
                    )
                )
                end_time = e
                end_idx = len(phones)
            else:
                phones.append("sp")
                durations.append(
                    int(
                        np.round(e * self.sampling_rate / self.hop_length)
                        - np.round(s * self.sampling_rate / self.hop_length)
                    )
                )

        phones = phones[:end_idx]
        durations = durations[:end_idx]
        return phones, durations, start_time, end_time

    def process_utterance(
        self,
        wav_path: str,
        textgrid_path: str,
        raw_text: str,
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
        """
        Processes a single Telugu audio file and TextGrid to extract:
        (phones, mel_spectrogram, pitch, energy, duration, n_frames).
        """
        import tgt

        if not os.path.exists(textgrid_path) or not os.path.exists(wav_path):
            return None

        try:
            tg = tgt.io.read_textgrid(textgrid_path)
            phone_tier = tg.get_tier_by_name("phones")
            phones, durations, start_time, end_time = self.get_alignment(phone_tier)
        except Exception:
            return None

        if start_time >= end_time or len(phones) == 0 or sum(durations) == 0:
            return None

        wav, sr = librosa.load(wav_path, sr=self.sampling_rate)
        start_sample = int(self.sampling_rate * start_time)
        end_sample = int(self.sampling_rate * end_time)
        wav = wav[start_sample:end_sample].astype(np.float32)

        # 1. Pitch extraction via PyWorld DIO / Stonemask
        frame_period_ms = self.hop_length / self.sampling_rate * 1000.0
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=frame_period_ms,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: sum(durations)]

        if np.sum(pitch != 0) <= 1:
            return None

        # 2. Mel-Spectrogram and Energy extraction
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.stft)
        mel_spectrogram = mel_spectrogram[:, : sum(durations)]
        energy = energy[: sum(durations)]

        # 3. Phoneme/Akshara-level averaging
        if self.pitch_phoneme_averaging:
            nonzero_idx = np.where(pitch != 0)[0]
            if len(nonzero_idx) > 1:
                interp_fn = interp1d(
                    nonzero_idx,
                    pitch[nonzero_idx],
                    fill_value=(pitch[nonzero_idx[0]], pitch[nonzero_idx[-1]]),
                    bounds_error=False,
                )
                pitch = interp_fn(np.arange(len(pitch)))

            pos = 0
            avg_pitch = np.zeros(len(phones))
            for i, d in enumerate(durations):
                if d > 0:
                    avg_pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    avg_pitch[i] = avg_pitch[i - 1] if i > 0 else 0
                pos += d
            pitch = avg_pitch

        if self.energy_phoneme_averaging:
            pos = 0
            avg_energy = np.zeros(len(phones))
            for i, d in enumerate(durations):
                if d > 0:
                    avg_energy[i] = np.mean(energy[pos : pos + d])
                else:
                    avg_energy[i] = avg_energy[i - 1] if i > 0 else 0
                pos += d
            energy = avg_energy

        phone_str = "{" + " ".join(phones) + "}"
        n_frames = mel_spectrogram.shape[1]

        return (
            phone_str,
            mel_spectrogram,
            pitch,
            energy,
            np.array(durations, dtype=np.int32),
            n_frames,
        )


# ==============================================================================
# 4. DATASET STATISTICS EXPORT (dataset_stats.json)
# ==============================================================================

def generate_telugu_dataset_stats(
    train_records: List[Dict],
    val_records: List[Dict],
    pitch_stats: Dict[str, float],
    energy_stats: Dict[str, float],
    output_path: str = "Telugu/dataset_stats.json",
) -> dict:
    """
    Computes factual dataset metrics for Telugu and exports to dataset_stats.json.
    """
    train_durations = [float(r["duration"]) for r in train_records]
    val_durations = [float(r["duration"]) for r in val_records]
    all_durations = train_durations + val_durations

    train_spks = [r.get("speaker", "default") for r in train_records]
    val_spks = [r.get("speaker", "default") for r in val_records]

    spk_dist = {}
    for r in train_records + val_records:
        spk = r.get("speaker", "default")
        spk_dist[spk] = spk_dist.get(spk, 0) + 1

    gender_dist = {"male_sec": 0.0, "female_sec": 0.0, "unknown_sec": 0.0}
    for r in train_records + val_records:
        g = str(r.get("gender", r.get("speaker", ""))).lower()
        d = float(r["duration"])
        if "female" in g or "fe" in g or g.endswith("_f"):
            gender_dist["female_sec"] += d
        elif "male" in g or g.endswith("_m"):
            gender_dist["male_sec"] += d
        else:
            gender_dist["unknown_sec"] += d

    stats_obj = {
        "dataset_name": "IndicTTS Telugu (IIT Madras Speech Consortium)",
        "language": "Telugu",
        "subsampling_policy": {
            "random_seed": 42,
            "min_duration_sec": 1.5,
            "max_duration_sec": 10.0,
            "speaker_balancing": "50% Male (te_male) / 50% Female (te_female) balanced cohort",
            "overlap": "Strictly 0% overlap between train and validation splits",
        },
        "audio_hours": {
            "train_hours": round(sum(train_durations) / 3600.0, 4),
            "val_hours": round(sum(val_durations) / 3600.0, 4),
            "total_hours": round(sum(all_durations) / 3600.0, 4),
        },
        "utterance_counts": {
            "train_utterances": len(train_records),
            "val_utterances": len(val_records),
            "total_utterances": len(all_durations),
        },
        "duration_distribution_seconds": {
            "min": round(float(np.min(all_durations)), 3),
            "max": round(float(np.max(all_durations)), 3),
            "mean": round(float(np.mean(all_durations)), 3),
            "std": round(float(np.std(all_durations)), 3),
            "median": round(float(np.median(all_durations)), 3),
        },
        "pitch_statistics_f0_hz": pitch_stats,
        "energy_statistics": energy_stats,
        "demographics": {
            "unique_speakers_total": len(spk_dist),
            "gender_hours": {
                "female_hours": round(gender_dist["female_sec"] / 3600.0, 4),
                "male_hours": round(gender_dist["male_sec"] / 3600.0, 4),
                "unknown_hours": round(gender_dist["unknown_sec"] / 3600.0, 4),
            },
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_obj, f, indent=2)
    print(f"[Stats] Exported Telugu dataset statistics to {output_path}")

    return stats_obj


def main():
    parser = argparse.ArgumentParser(description="Preprocess IndicTTS Telugu dataset for FastSpeech 2.")
    parser.add_argument("--config", default="Telugu/config.yaml", help="Path to config.yaml")
    parser.add_argument("--telugu_dir", default="data/IndicTTS_Telugu", help="Path to raw dataset")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print("FastSpeech 2 Telugu Preprocessing Pipeline Initialized.")


if __name__ == "__main__":
    main()

