# Architectural & Engineering Decisions Log

This document records key technical decisions, trade-offs, and rejected alternatives for the FastSpeech 2 reference implementation in English and Telugu.

---

## Stage 6: Technical Report Skeleton & Checkpoint Storage Strategy

### Decision 18: Factual Completion with Explicit Empirical Placeholders
- **Chosen**: Fully author all technical, mathematical, configuration, and reproducibility sections in `Report/Technical_Report.md`, while placing explicit `[TODO FOR USER]` prompts for subjective audio analysis, loss curve interpretations, and plot observations.
- **Why**: Adheres strictly to the user's constraint: *"Mark observations and analysis as TODO for me. Do not write them."* and Hard Rule 2 (*"Never fabricate anything"*).
- **Alternatives Rejected**:
  - *Fabricating plausible-sounding listening impressions*: Violates academic integrity and Hard Rule 2.

### Decision 19: Dual Checkpoint Distribution (Git LFS + External Links)
- **Chosen**: Document both Git LFS tracking commands and public external hosting guidelines (Google Drive / Hugging Face / Kaggle).
- **Why**: Satisfies Section 4b of the assignment specification, which notes: *"If model checkpoints exceed GitHub's file-size limits, use Git LFS or provide an accessible external download link in the README."*


## Stage 5: Inference Architecture, Variance Adaptor Plotting & Loss Curves

### Decision 15: Per-Sentence Variance Adaptor Visualization Format
- **Chosen**:
  - Duration: Discrete bar chart showing frames/phoneme aligned to linguistic tokens on the x-axis.
  - Pitch: Continuous $F_0$ curve (Hz) over mel frames.
  - Energy: Continuous amplitude curve over mel frames.
- **Why**: Aligns with Section 1 and Section 4 of the PDF specification, which requires applicants to "analyse the duration, pitch and energy predictions of the variance adaptor for the given test samples."
- **Alternatives Rejected**:
  - *Unified single plot*: Stacking all three contours on one small figure creates visual clutter and illegible token labels for long sentences like E1 (28-letter word) or T4.

### Decision 16: Neural Vocoder Selection per Language
- **Chosen**:
  - English: HiFi-GAN LJSpeech generator (`hifigan/generator_LJSpeech.pth.tar`).
  - Telugu: HiFi-GAN Universal generator (`hifigan/generator_universal.pth.tar`).
- **Why**: The Universal generator was trained across multi-speaker, diverse phonetic distributions, generalizing far better to non-English Indic phonetics than an English-only single-speaker vocoder.
- **Alternatives Rejected**:
  - *MelGAN*: Lower audio fidelity and prone to metallic phase artifacts compared to HiFi-GAN.

### Decision 17: Strict Non-Fabrication of Loss Curves & Audio (Hard Rule 2)
- **Chosen**: Implement `plot_losses.py` to parse real TensorBoard event binaries without generating fake or synthetic plots before real training is executed.
- **Why**: Hard Rule 2 strictly prohibits mock/fabricated loss values or placeholder audio files. The pipeline is fully wired to ingest real training logs and generate genuine submission plots upon model execution on Kaggle/Colab.


## Stage 4: Training Pipeline, Optimization & Checkpointing

### Decision 12: Unified Single `config.yaml` with Modular Sub-Dictionaries
- **Chosen**: Structure `English/config.yaml` and `Telugu/config.yaml` with top-level `preprocess`, `model`, and `train` sections.
- **Why**: The PDF folder hierarchy mandates a single `config.yaml` in each language folder, while the base FastSpeech 2 repository expects three separate dictionaries `(preprocess_config, model_config, train_config)`. Providing top-level sections allows `train.py` and `inference.py` to seamlessly unpack the configurations without modifying the underlying model or dataset classes.
- **Alternatives Rejected**:
  - *Hardcoding paths and parameters inside Python code*: Destroys configurability and violates clean software engineering standards.
  - *Keeping three separate YAML files per language*: Violates the assignment PDF folder structure.

### Decision 13: Explicit Dual-Split TensorBoard Logging
- **Chosen**: Log training losses (`Train/Total_Loss`, `Train/Mel_Loss`, `Train/Duration_Loss`, etc.) and validation losses (`Validation/Total_Loss`, `Validation/Mel_Loss`, etc.) under distinct hierarchical tags.
- **Why**: Allows TensorBoard to display train and validation metrics side-by-side or overlaid, enabling immediate detection of overfitting or divergence.
- **Alternatives Rejected**:
  - *Logging only train loss*: Prevents monitoring model generalization and makes it impossible to detect overfitting.

### Decision 14: Dual Checkpoint Retention Strategy
- **Chosen**: Save permanent step checkpoints (`<step>.pth.tar`) every `save_step` and continuously update a rolling `latest.pth.tar`.
- **Why**: Protects against unexpected training preemption on preemptible cloud GPUs (Google Colab disconnections or Kaggle 12-hour session limits), while retaining milestones for historical evaluation.


## Stage 3: Telugu Frontend, Normalization & Alignment Strategy

### Decision 8: Native Telugu Grapheme Set vs. External G2P
- **Chosen**: Native Telugu Unicode characters (graphemes) in `text/symbols.py`.
- **Why**: Telugu has a shallow, phonemic orthography where characters map directly to acoustic sounds with predictable regularity. External G2P libraries (`epitran`, `indic-nlp`) introduce fragile dependencies that easily crash on corrupted Unicode strings or non-standard loanwords. FastSpeech 2's 1D convolutional layers in the encoder reliably learn local conjunct relationships (`క + ్ + ష`) directly from graphemes.
- **Alternatives Rejected**:
  - *Transliteration to ASCII via Unidecode*: Destroys phonetic contrast and tone, making Indic speech synthesis completely unintelligible.
  - *Third-party IPA / G2P phone tables*: High risk of runtime crashes on the corrupted PDF test sentences.

### Decision 9: Self-Contained Number, Date & Time Normalization
- **Chosen**: Custom, zero-dependency regex engine (`text/telugu_numbers.py`).
- **Why**: Non-standard words (NSW) in T4 (`2026`, `12వ తేదీ`, `10:45 గంటలకు`) require specific morphological inflections in Telugu (e.g. ordinal suffix `వ` $\rightarrow$ `పన్నెండవ`, time suffix $\rightarrow$ `గంటలకు`). Third-party packages like `indic-num2words` only handle bare integers, failing on ordinals and clock times.
- **Alternatives Rejected**:
  - *Using English number normalization*: Speaks English numbers in the middle of Telugu sentences, breaking linguistic consistency.

### Decision 10: Dual-Pass Inference Strategy for Test Sentences (Hard Rule 4)
- **Chosen**: Provide both strict verbatim synthesis and cleaned/repaired synthesis for evaluation.
- **Why**: Hard Rule 4 mandates copying test sentences verbatim. However, sentences T1–T4 contain broken Unicode conjuncts (e.g. `విశ్లషే ించారు`, `నిశ్శబ్ంద`). Evaluating both verbatim and repaired inputs provides an empirical basis for the Technical Report's error analysis, comparing variance adaptor behaviour on corrupted vs. well-formed text.
- **Alternatives Rejected**:
  - *Silently modifying the test sentences*: Violates Hard Rule 4 ("Copy the 10 test sentences from the PDF verbatim... Preserve the sentence text as provided").
  - *Crashing on corrupt characters*: Prevents generating test audios and variance plots required by Section 4 of the PDF.

### Decision 11: 50% Male / 50% Female Speaker Cohort for IndicTTS
- **Chosen**: Equal allocation of male (`te_male`) and female (`te_female`) recordings across train (~5h) and val (~1h) splits.
- **Why**: IndicTTS Telugu provides roughly 4.47 hours male and 4.27 hours female speech. A balanced 1:1 mixture ensures the model encounters both pitch registers without gender skew.


## Stage 2: English Svarah Preprocessing & Alignment

### Decision 4: Svarah Duration Filtering Range [1.5s, 10.0s]
- **Chosen**: Discard utterances shorter than 1.5 seconds and longer than 10.0 seconds.
- **Why**: Utterances below 1.5 seconds often consist of short filler words, clicks, or truncated syllables lacking complete intonation contours. Utterances above 10.0 seconds trigger severe quadratic memory bloat ($O(T^2)$) during Transformer self-attention and cause alignment drift/failures in MFA.
- **Alternatives Rejected**:
  - *No duration filtering*: Results in CUDA out-of-memory errors on GPU and noisy phoneme alignments from truncated audio clips.
  - *Narrow filtering [3.0s, 6.0s]*: Discards too many valid conversational sentences, making it difficult to reach the required ~5 hours of training data.

### Decision 5: Speaker Cohort Balancing & Disjoint Splitting
- **Chosen**: Round-robin utterance selection across speakers with gender parity (50% Male, 50% Female), fixed with deterministic random seed 42, ensuring strictly 0% overlap between train and validation.
- **Why**: Svarah comprises 117 speakers. Training a non-autoregressive acoustic model without speaker ID conditioning on 117 sparsely sampled speakers results in an averaged, blurry acoustic distribution. Speaker balancing prevents dominant speakers from biasing pitch/energy statistics while maintaining dialectal representation.
- **Alternatives Rejected**:
  - *Random unstratified sampling*: Leads to severe speaker imbalance where speakers with longer monologues dominate the training distribution.
  - *Single-speaker filtering*: Reduces total usable audio below the mandatory 6 hours (5h train + 1h val).

### Decision 6: Audio Resampling to 22,050 Hz
- **Chosen**: Resample all input speech to 22,050 Hz using high-quality sinc/polyphase resampling (`librosa.resample`).
- **Why**: Matches the native sample rate of the pretrained HiFi-GAN neural vocoder, eliminating sample rate mismatch artifacts during inference.
- **Alternatives Rejected**:
  - *Keeping raw 16 kHz or 48 kHz*: Requires retraining the neural vocoder from scratch, which is outside the scope of FastSpeech 2 training and computationally prohibitive.

### Decision 7: Phoneme-Level vs Frame-Level Pitch & Energy Averaging
- **Chosen**: Phoneme-level averaging for both pitch ($F_0$) and energy.
- **Why**: FastSpeech 2 predicts duration at the phoneme level first, then expands phonemes to mel frames via the Length Regulator. Aligning pitch and energy predictors to phoneme boundaries ensures that each linguistic unit receives a coherent acoustic target during inference, preventing frame-level jitter or pitch contour discontinuities.
- **Alternatives Rejected**:
  - *Frame-level prediction*: Increases variance predictor difficulty and causes mismatch when predicted durations differ from ground truth.


## Stage 1: Repository Structure & Base Selection

### Decision 1: Base Implementation Selection
- **Chosen**: `ming024/FastSpeech2` PyTorch implementation.
- **Why**: It is the gold-standard open-source reference for FastSpeech 2 with explicit pitch/energy variance predictors, Montreal Forced Aligner (MFA) TextGrid support, PyWorld pitch extraction, and pre-integrated HiFi-GAN vocoder.
- **Alternatives Rejected**:
  - *Writing from scratch*: Violates Rule 1 ("Do not rewrite from scratch. Modify minimally and document every change").
  - *ESPnet / Fairseq*: Excessively complex monolithic frameworks with heavy abstract wrappers that hinder pedagogical clarity and make code inspection difficult for learning.

### Decision 2: Directory Architecture & Root Import Strategy
- **Chosen**: Retain core model libraries (`model/`, `transformer/`, `utils/`, `audio/`, `text/`, `hifigan/`) in the root directory, with lightweight per-language orchestration directories (`English/` and `Telugu/`) conforming strictly to the assignment directory layout.
- **Why**: Keeps code DRY (Don't Repeat Yourself) while satisfying the assignment's explicit directory constraints (`English/train.py`, `Telugu/train.py`, etc.).
- **Alternatives Rejected**:
  - *Duplicating all PyTorch model code into both `English/` and `Telugu/`*: Results in 2x code bloat, divergence of core bug fixes, and maintenance overhead.
  - *Single script with language flag*: Violates the assignment PDF folder structure which explicitly requires `English/train.py` and `Telugu/train.py`.

### Decision 3: Python Environment & Modern Dependency Management
- **Chosen**: Target Python 3.10 / 3.11 with PyTorch 2.x and backwards-compatible audio/alignment libraries (`tgt`, `pyworld`, `g2p-en`).
- **Why**: Allows reproducible execution on local workstations, modern Google Colab runtimes, and Kaggle GPU kernels without build errors caused by deprecated C-extensions.
- **Alternatives Rejected**:
  - *Sticking to 2020 pinned versions (`torch==1.7.0`)*: Fails on any modern machine or cloud GPU platform due to lack of wheel binaries for modern CUDA drivers and modern Python versions.

