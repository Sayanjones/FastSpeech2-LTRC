# Changes Log

This document tracks all modifications made to the reference FastSpeech 2 codebase for both English and Telugu implementations, explaining why each change was made and which component it affects.

---

## Documentation, Reproducibility & Report Skeleton (Stage 6)

### 1. Master Documentation (`README.md`)
- **Files Created**: `README.md`
- **Why**: Provides the end-to-end user guide, environment setup, dataset acquisition, reproduction commands, Git LFS checkpoint tracking, and external checkpoint download link instructions per Section 4b of the specification.

### 2. Technical Report Skeleton (`Report/Technical_Report.md`)
- **Files Created**: `Report/Technical_Report.md`
- **Why**: Structures the final assignment report:
  - Thoroughly fills in all factual, objective sections (architecture diagrams, equations, configurations, hyperparameter comparison tables, dataset subsampling rules, Unicode anomaly audits, dependencies, and reproduction commands).
  - Strictly marks all subjective analysis and empirical evaluation sections (convergence interpretation, speech quality observations, variance adaptor plot pattern analyses, and English vs Telugu comparative evaluations) as `[TODO FOR USER]` per project instructions and Hard Rule 2.


## Initial Setup & Repository Restructuring (Stage 1)

### 1. Root & Modular Directory Layout
- **Files Affected**:
  - `English/`: Dedicated scripts and configuration for English TTS (`train.py`, `preprocess.py`, `inference.py`, `config.yaml`, `checkpoints/`).
  - `Telugu/`: Dedicated scripts and configuration for Telugu TTS (`train.py`, `preprocess.py`, `inference.py`, `config.yaml`, `checkpoints/`).
  - `Results/`: Standard output directories for loss plots (`training_loss.png`, `validation_loss.png`), audio files (`test_generated_audios/`), and variance adaptor plots (`duration/`, `pitch/`, `energy/`) for both languages.
  - `Report/`: Technical report directory.
- **Why**: Follows the exact submission structure specified in Section 4 of the assignment specification.
- **Base Components Preserved**: Core modules (`model/`, `transformer/`, `audio/`, `utils/`, `hifigan/`, `text/`, `preprocessor/`) are maintained at the repository root as shared architectural building blocks.

### 6. PyTorch 2.6+ Checkpoint Unpickling Modernization
- **Files Affected**: `English/inference.py`, `Telugu/inference.py`, `utils/model.py`
- **Why**: PyTorch 2.6+ changed the default behavior of `torch.load` to `weights_only=True`, which raises `_pickle.UnpicklingError` when loading composite training checkpoints containing optimizer state dictionaries or numpy scaler objects.
- **Update**: Explicitly passed `weights_only=False` to `torch.load` when restoring model checkpoints and pretrained HiFi-GAN vocoders.

---

## Documentation, Reproducibility & Report Skeleton (Stage 6)

### 1. Master Documentation (`README.md`)
- **Files Created**: `README.md`
- **Why**: Provides the end-to-end user guide, environment setup, dataset acquisition, reproduction commands, Git LFS checkpoint tracking, and external checkpoint download link instructions per Section 4b of the specification.

### 2. Technical Report Skeleton (`Report/Technical_Report.md`)
- **Files Created**: `Report/Technical_Report.md`
- **Why**: Structures the final assignment report:
  - Thoroughly fills in all factual, objective sections (architecture diagrams, equations, configurations, hyperparameter comparison tables, dataset subsampling rules, Unicode anomaly audits, dependencies, and reproduction commands).
  - Strictly marks all subjective analysis and empirical evaluation sections (convergence interpretation, speech quality observations, variance adaptor plot pattern analyses, and English vs Telugu comparative evaluations) as `[TODO FOR USER]` per project instructions and Hard Rule 2.


## English Changes (Stage 2)

### 1. English Configuration (`English/config.yaml`)
- **Files Created**: `English/config.yaml`
- **Why**: Unifies dataset paths, subsampling thresholds (1.5s to 10.0s), resampling parameters (22,050 Hz), STFT filter/hop settings (1024/256), model dimensions (256 hidden, 4 encoder/decoder layers), variance predictor settings, and training hyperparameters into a single, self-contained configuration file conforming to the PDF deliverable structure.

### 2. Svarah Subsampling & Preprocessing Pipeline (`English/preprocess.py`)
- **Files Created**: `English/preprocess.py`
- **Why**: Implements the end-to-end preprocessing pipeline for English Svarah:
  - **Subsampling**: Duration filtering ($1.5\text{s} \le \text{duration} \le 10.0\text{s}$) to remove non-speech/truncated clips and prevent Transformer OOM errors.
  - **Speaker Balancing**: Round-robin speaker cohort selection enforcing gender parity (50% Male, 50% Female).
  - **Zero-Overlap Split**: Deterministic seed (`seed = 42`) generating ~5.0 hours train and ~1.0 hour validation with strictly 0% utterance overlap.
  - **Resampling**: Standardizes all audio to 22,050 Hz 16-bit mono PCM.
  - **MFA Alignment Preparation**: Builds `mfa_corpus/<speaker>/` directory structure with audio and `.lab` transcripts for Montreal Forced Aligner.
  - **Variance Feature Extraction**: PyWorld DIO/Stonemask for continuous $F_0$, STFT magnitude L2-norm for energy, phoneme-level averaging, and StandardScaler normalization.
  - **Statistics Export**: Computes and exports `English/dataset_stats.json`.


---

## Documentation, Reproducibility & Report Skeleton (Stage 6)

### 1. Master Documentation (`README.md`)
- **Files Created**: `README.md`
- **Why**: Provides the end-to-end user guide, environment setup, dataset acquisition, reproduction commands, Git LFS checkpoint tracking, and external checkpoint download link instructions per Section 4b of the specification.

### 2. Technical Report Skeleton (`Report/Technical_Report.md`)
- **Files Created**: `Report/Technical_Report.md`
- **Why**: Structures the final assignment report:
  - Thoroughly fills in all factual, objective sections (architecture diagrams, equations, configurations, hyperparameter comparison tables, dataset subsampling rules, Unicode anomaly audits, dependencies, and reproduction commands).
  - Strictly marks all subjective analysis and empirical evaluation sections (convergence interpretation, speech quality observations, variance adaptor plot pattern analyses, and English vs Telugu comparative evaluations) as `[TODO FOR USER]` per project instructions and Hard Rule 2.


## Telugu Changes (Stage 3)

### 1. Telugu Alphabet & Unicode Symbols (`text/symbols.py`, `text/telugu.py`)
- **Files Modified / Created**: `text/symbols.py`, `text/telugu.py`
- **Why**: FastSpeech 2's text encoder requires an integer embedding table. Extended the vocabulary with native Telugu Unicode characters (independent vowels, consonants, dependent mātrās, virama, anusvara, visarga) from U+0C00–U+0C7F.
- **Benefit**: Preserves Telugu orthography without lossy ASCII transliteration, allowing native tokenization and embedding.

### 2. Telugu Number, Ordinal & Time Normalization (`text/telugu_numbers.py`)
- **Files Created**: `text/telugu_numbers.py`
- **Why**: Handles non-standard word (NSW) expansion for T4:
  - Years/Cardinals: `2026` -> *రెండు వేల ఇరవై ఆరు*
  - Ordinal dates: `12వ తేదీ` -> *పన్నెండవ తేదీ*
  - Clock times: `10:45 గంటలకు` -> *పది నలభై ఐదు గంటలకు*

### 3. Conjunct Repair Engine & Cleaners (`text/cleaners.py`, `text/telugu.py`)
- **Files Modified**: `text/cleaners.py`
- **Why**: Added `telugu_cleaners` and `telugu_cleaners_with_repair` to handle real-world PDF extraction glitches:
  - Reattaches orphaned dependent vowel signs (`వచ్చ ింది` -> `వచ్చింది`).
  - Repairs displaced anusvaras following viramas (`నిశ్శబ్ంద గా` -> `నిశ్శబ్దంగా`, `సిద్ంధ గా` -> `సిద్ధంగా`).
  - Corrects swapped vowel modifiers (`విశ్లషే ించారు` -> `విశ్లేషించారు`).

### 4. Telugu Configuration & Preprocessing (`Telugu/config.yaml`, `Telugu/preprocess.py`)
- **Files Created**: `Telugu/config.yaml`, `Telugu/preprocess.py`
- **Why**: Implements the complete IndicTTS Telugu pipeline:
  - Subsamples ~5.0 hours train / ~1.0 hour validation with strictly 0% overlap and seed 42.
  - Enforces 50% male and 50% female speaker balance.
  - Automatically generates a 1-to-1 Telugu grapheme lexicon (`telugu_lexicon.txt`) for MFA alignment.
  - Extracts F0 (PyWorld DIO/Stonemask), STFT energy, mel-spectrograms, and computes `Telugu/dataset_stats.json`.

---

## Documentation, Reproducibility & Report Skeleton (Stage 6)

### 1. Master Documentation (`README.md`)
- **Files Created**: `README.md`
- **Why**: Provides the end-to-end user guide, environment setup, dataset acquisition, reproduction commands, Git LFS checkpoint tracking, and external checkpoint download link instructions per Section 4b of the specification.

### 2. Technical Report Skeleton (`Report/Technical_Report.md`)
- **Files Created**: `Report/Technical_Report.md`
- **Why**: Structures the final assignment report:
  - Thoroughly fills in all factual, objective sections (architecture diagrams, equations, configurations, hyperparameter comparison tables, dataset subsampling rules, Unicode anomaly audits, dependencies, and reproduction commands).
  - Strictly marks all subjective analysis and empirical evaluation sections (convergence interpretation, speech quality observations, variance adaptor plot pattern analyses, and English vs Telugu comparative evaluations) as `[TODO FOR USER]` per project instructions and Hard Rule 2.


---

## Documentation, Reproducibility & Report Skeleton (Stage 6)

### 1. Master Documentation (`README.md`)
- **Files Created**: `README.md`
- **Why**: Provides the end-to-end user guide, environment setup, dataset acquisition, reproduction commands, Git LFS checkpoint tracking, and external checkpoint download link instructions per Section 4b of the specification.

### 2. Technical Report Skeleton (`Report/Technical_Report.md`)
- **Files Created**: `Report/Technical_Report.md`
- **Why**: Structures the final assignment report:
  - Thoroughly fills in all factual, objective sections (architecture diagrams, equations, configurations, hyperparameter comparison tables, dataset subsampling rules, Unicode anomaly audits, dependencies, and reproduction commands).
  - Strictly marks all subjective analysis and empirical evaluation sections (convergence interpretation, speech quality observations, variance adaptor plot pattern analyses, and English vs Telugu comparative evaluations) as `[TODO FOR USER]` per project instructions and Hard Rule 2.


## Inference Pipeline & Results Generation (Stage 5)

### 1. English Inference (`English/inference.py`)
- **Files Created**: `English/inference.py`
- **Why**: Synthesizes speech for all 5 English test sentences (`E1`, `E2a`, `E2b`, `E3`, `E4`) verbatim per Section 3 of the PDF.
- **Functionality**:
  - Forward pass through FastSpeech 2 variance adaptor.
  - Vocodes predicted mel-spectrogram using HiFi-GAN LJSpeech generator (`hifigan/generator_LJSpeech.pth.tar`).
  - Saves synthesized WAV audio to `Results/English/test_generated_audios/{id}.wav`.
  - Automatically exports per-sentence variance adaptor plots:
    * `Results/English/variance_adaptor_plots/duration/{id}_duration.png`
    * `Results/English/variance_adaptor_plots/pitch/{id}_pitch.png`
    * `Results/English/variance_adaptor_plots/energy/{id}_energy.png`

### 2. Telugu Inference (`Telugu/inference.py`)
- **Files Created**: `Telugu/inference.py`
- **Why**: Synthesizes speech for all 5 Telugu test sentences (`T1`, `T2a`, `T2b`, `T3`, `T4`).
- **Functionality**:
  - Implements dual-mode synthesis (`--mode verbatim` and `--mode repaired`) to allow direct empirical error analysis of scrambled vs. cleaned Indic conjuncts.
  - Vocodes predicted mel-spectrogram using HiFi-GAN Universal generator (`hifigan/generator_universal.pth.tar`).
  - Saves audio to `Results/Telugu/test_generated_audios/{id}.wav`.
  - Automatically exports per-sentence variance adaptor plots into `Results/Telugu/variance_adaptor_plots/{duration,pitch,energy}/`.

### 3. Loss Plotting Engine (`plot_losses.py`)
- **Files Created**: `plot_losses.py`
- **Why**: Reads real TensorBoard event files (`events.out.tfevents.*`) and generates multi-panel figures for:
  - `Results/<Language>/training_loss.png` (Total, Mel, Duration, Pitch, Energy loss curves).
  - `Results/<Language>/validation_loss.png` (Validation loss curves over evaluation steps).
  - Enforces Hard Rule 2 by rejecting mock/placeholder data and operating strictly on real event logs.

### 4. HiFi-GAN Weights Unpacking
- **Files Modified**: Unzipped `hifigan/generator_LJSpeech.pth.tar.zip` and `hifigan/generator_universal.pth.tar.zip`.
- **Why**: `utils/model.py:get_vocoder()` expects uncompressed `.pth.tar` checkpoints.


### 1. Unified Configuration Architecture (`English/config.yaml`, `Telugu/config.yaml`)
- **Files Updated**: `English/config.yaml`, `Telugu/config.yaml`
- **Why**: Consolidated the reference implementation's three fragmented config files (`preprocess.yaml`, `model.yaml`, `train.yaml`) into a single, comprehensive `config.yaml` per language, meeting Section 4's exact directory specification while maintaining full compatibility with the underlying PyTorch modules.

### 2. Multi-Task Loss Training Scripts (`English/train.py`, `Telugu/train.py`)
- **Files Created**: `English/train.py`, `Telugu/train.py`
- **Why**: Implements parallel, multi-task training for FastSpeech 2:
  - Multi-task loss tracking: Total loss, Mel-spectrogram L1 loss, PostNet mel loss, Duration L2 loss, Pitch ($F_0$) L2 loss, and Energy L2 loss.
  - TensorBoard logging: Explicit scalar logging under `Train/` and `Validation/` tags.
  - Checkpoint management: Periodic checkpoint saving every `save_step` (`<step>.pth.tar`) and continuous saving of `latest.pth.tar`.
  - Resumption: `--restore_step <int>` restores model weights and ScheduledOptim optimizer learning rate state.



