# FastSpeech 2 TTS: English (Svarah) & Telugu (IndicTTS)

A complete, pedagogical reference implementation of **FastSpeech 2: Fast and High-Quality End-to-End Text to Speech** trained independently for:
1. **English**: Subsampled from the **Svarah** benchmark (AI4Bharat Indic-accented English, ~5.0 h train / ~1.0 h validation, disjoint).
2. **Telugu**: Subsampled from the **IndicTTS Telugu** dataset (IIT Madras Speech Consortium, ~5.0 h train / ~1.0 h validation, disjoint).

This repository conforms strictly to the submission structure and requirements specified in the assignment document.

---

## Repository Structure

```text
.
├── English/
│   ├── config.yaml          # Unified English configuration (preprocess, model, train)
│   ├── preprocess.py        # Svarah subsampling, resampling, MFA prep, F0/energy extraction
│   ├── train.py             # Multi-task training loop, TensorBoard, checkpoints, resume
│   ├── inference.py         # Synthesizes test sentences E1–E4, exports variance plots
│   └── checkpoints/         # Model weights directory (.gitkeep, *.pth.tar)
├── Telugu/
│   ├── config.yaml          # Unified Telugu configuration (preprocess, model, train)
│   ├── preprocess.py        # IndicTTS subsampling, grapheme lexicon, F0/energy extraction
│   ├── train.py             # Multi-task training loop, TensorBoard, checkpoints, resume
│   ├── inference.py         # Synthesizes test sentences T1–T4 (verbatim & repaired)
│   └── checkpoints/         # Model weights directory (.gitkeep, *.pth.tar)
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/       # E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav
│   │   └── variance_adaptor_plots/
│   │       ├── duration/                # E1_duration.png ... E4_duration.png
│   │       ├── pitch/                   # E1_pitch.png ... E4_pitch.png
│   │       └── energy/                  # E1_energy.png ... E4_energy.png
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/       # T1.wav, T2a.wav, T2b.wav, T3.wav, T4.wav
│       └── variance_adaptor_plots/
│           ├── duration/                # T1_duration.png ... T4_duration.png
│           ├── pitch/                   # T1_pitch.png ... T4_pitch.png
│           └── energy/                  # T1_energy.png ... T4_energy.png
├── Report/
│   ├── Technical_Report.md  # Detailed technical report skeleton (factual sections filled)
│   └── Technical_Report.pdf # Final compiled PDF report
├── model/                   # Core PyTorch FastSpeech 2 model, Variance Adaptor, Loss, Optimizer
├── transformer/             # Feed-Forward Transformer (FFT) blocks & Multi-Head Self-Attention
├── audio/                   # STFT audio transformations and mel-spectrogram extractors
├── utils/                   # Model loading, vocoder wrappers, tensor utilities
├── text/                    # Grapheme/phoneme tokenizers, CMUDict, Telugu cleaners & numbers
├── hifigan/                 # Pretrained neural vocoders (LJSpeech & Universal)
├── dataset.py               # Dynamic PyTorch Dataset & length-based batch collation
├── evaluate.py              # Validation evaluation loop
├── test_sentences.py        # Verbatim test sentences registry & Unicode anomaly audit
├── plot_losses.py           # Real TensorBoard log parser -> training_loss.png & validation_loss.png
├── requirements.txt         # Pinned modern dependencies
├── CHANGES.md               # Log of all modified/created files and justifications
└── DECISIONS.md             # Architectural decisions log & rejected alternatives
```

---

## Test Sentences (Verbatim)

All 10 test utterances from the assignment specification are copied verbatim (including intentional spacing anomalies and PDF conjunct extraction artifacts) and registered in [`test_sentences.py`](test_sentences.py).

### English Test Sentences
| ID | Sentence | Phonetic / Prosodic Challenge |
| :--- | :--- | :--- |
| **E1** | `The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.` | Extremely long compound word (28 letters); tests OOV stress handling in G2P. |
| **E2a** | `The orchestra's conduct during the ceremony was praised by the audience.` | Heteronym: `"conduct"` used as a **noun** (`/ˈkɒndʌkt/`, stress on 1st syllable). |
| **E2b** | `The researchers will conduct an experiment to evaluate the proposed method.` | Heteronym: `"conduct"` used as a **verb** (`/kənˈdʌkt/`, stress on 2nd syllable). |
| **E3** | `The rhyth m of the orchestra changed unexpectedly.` | **Odd Spacing Glitch**: Contains an artificial space splitting `"rhythm"` into `"rhyth m"`. |
| **E4** | `The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.` | Contains time (`10:45 AM`), date (`12 November 2026`), and code-mixed Telugu word (`"andaroo"`). |

### Telugu Test Sentences & Unicode Conjunct Analysis
| ID | Sentence | Unicode Extraction Artifacts & Linguistic Notes |
| :--- | :--- | :--- |
| **T1** | `న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు.` | 1. Spacing in loanword (`న్యూ రోఫార్మ కోలాజికల్`).<br>2. Merged boundary (`పరిశోధనలోనూతన`).<br>3. Scrambled conjunct: `విశ్లషే ించారు.` has orphaned vowel sign `ి` after space. Intended: **విశ్లేషించారు**. |
| **T2a** | `తాడుపై అతని పట్టుచాలా బలంగా ఉంది.` | Merged compound: `"పట్టుచాలా"` (*పట్టు* [grip] + *చాలా* [very]). |
| **T2b** | `ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది.` | Artificial space splitting geminate verb: `"వచ్చ ింది."`. Intended: **వచ్చింది**. |
| **T3** | `పులి అడ విలో నిశ్శబ్ంద గా నడిచింది.` | 1. Artificial space in `"అడ విలో"`.<br>2. Scrambled sequence: `"నిశ్శబ్ంద గా"` has virama (`్`) + anusvara (`ం`) before `ద`. Intended: **నిశ్శబ్దంగా**. |
| **T4** | `ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి.` | 1. Scrambled conjunct: `"సిద్ంధ గా"`. Intended: **సిద్ధంగా**.<br>2. Spoken expansion: `2026` $\rightarrow$ *రెండు వేల ఇరవై ఆరు*, `12వ` $\rightarrow$ *పన్నెండవ*, `10:45` $\rightarrow$ *పది నలభై ఐదు*. |

---

## Installation & Environment Setup

### 1. Python Environment
Target environment: **Python 3.10+ / 3.11+**, **PyTorch >= 2.0.0** with CUDA GPU support.

```bash
# Clone repository
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>

# Create clean virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Montreal Forced Aligner (MFA) Setup (via Conda)
```bash
conda install -c conda-forge montreal-forced-aligner -y
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
```

---

## Dataset Acquisition & Subsampling Policy

### English: Svarah Benchmark (`ai4bharat/Svarah`)
- **Source**: Hugging Face dataset [`ai4bharat/Svarah`](https://huggingface.co/datasets/ai4bharat/Svarah).
- **Access**: This is a gated dataset. You must visit the dataset page, log in, and accept the license terms.
- **Subsampling Policy**:
  - Filter: Duration constrained to $[1.5\text{s}, 10.0\text{s}]$.
  - Speaker Balance: Round-robin cohort across speakers enforcing 50% Male / 50% Female parity.
  - Deterministic Split: `seed = 42`.
  - Disjointness: $\text{Train} \approx 5.0\text{ hours}$, $\text{Validation} \approx 1.0\text{ hour}$, $0\%$ utterance overlap.

### Telugu: IndicTTS Telugu Dataset (IIT Madras)
- **Source**: IIT Madras Speech Consortium / Hugging Face [`SPRINGLab/IndicTTS_Telugu`](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu).
- **Subsampling Policy**:
  - Filter: Duration constrained to $[1.5\text{s}, 10.0\text{s}]$.
  - Speaker Balance: Equal allocation between `te_male` (~4.47h available) and `te_female` (~4.27h available).
  - Deterministic Split: `seed = 42`.
  - Disjointness: $\text{Train} \approx 5.0\text{ hours}$, $\text{Validation} \approx 1.0\text{ hour}$, $0\%$ utterance overlap.

---

## End-to-End Execution Guide

### Step 1: Preprocessing

```bash
# 1. English Preprocessing (downloads via HF token, filters, resamples to 22.05 kHz, extracts F0/energy)
python English/preprocess.py --use_hf --config English/config.yaml

# 2. Telugu Preprocessing (builds grapheme lexicon, filters, resamples, extracts F0/energy)
python Telugu/preprocess.py --config Telugu/config.yaml
```

Each script automatically writes `English/dataset_stats.json` and `Telugu/dataset_stats.json`.

---

### Step 2: Training

```bash
# Train English FastSpeech 2 model
python English/train.py --config English/config.yaml

# Resume English training from checkpoint:
python English/train.py --config English/config.yaml --restore_step 20000

# Train Telugu FastSpeech 2 model
python Telugu/train.py --config Telugu/config.yaml

# Monitor training and validation losses in real-time
tensorboard --logdir English/log --port 6006
tensorboard --logdir Telugu/log --port 6007
```

---

### Step 3: Inference & Test Audio Generation

Synthesizes the test sentences, vocodes to WAV via HiFi-GAN, and generates per-sentence duration, pitch, and energy plots:

```bash
# English Synthesis (E1, E2a, E2b, E3, E4)
python English/inference.py --config English/config.yaml --checkpoint English/checkpoints/latest.pth.tar

# Telugu Synthesis - Verbatim Pass (T1, T2a, T2b, T3, T4 per Hard Rule 4)
python Telugu/inference.py --config Telugu/config.yaml --checkpoint Telugu/checkpoints/latest.pth.tar --mode verbatim

# Telugu Synthesis - Cleaned/Repaired Pass (for comparative error analysis in Report)
python Telugu/inference.py --config Telugu/config.yaml --checkpoint Telugu/checkpoints/latest.pth.tar --mode repaired
```

Outputs are automatically saved into:
- `Results/<Language>/test_generated_audios/{id}.wav`
- `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/{id}_*.png`

---

### Step 4: Loss Curves Generation

Parses real TensorBoard binary logs and generates `training_loss.png` and `validation_loss.png`:

```bash
# Generate English Loss Plots
python plot_losses.py --log_dir English/log --output_dir Results/English --language English

# Generate Telugu Loss Plots
python plot_losses.py --log_dir Telugu/log --output_dir Results/Telugu --language Telugu
```

---

## Checkpoint Management (Git LFS & External Links)

PyTorch checkpoints (`*.pth.tar`) exceed standard GitHub file-size limits (100 MB). Per Section 4b of the assignment specification:

### Option 1: Git LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track checkpoint archives
git lfs track "*.pth.tar"
git add .gitattributes
git commit -m "Configure Git LFS for PyTorch checkpoints"

# Push checkpoints to GitHub
git add English/checkpoints/*.pth.tar Telugu/checkpoints/*.pth.tar
git commit -m "Add trained model checkpoints"
git push origin main
```

### Option 2: External Download Links (Google Drive / Hugging Face / Kaggle)
If model checkpoints exceed your repository's Git LFS bandwidth/quota, host the weights externally and provide accessible public download links below:

- **English Model Checkpoint (`English_FastSpeech2_latest.pth.tar`)**:
  `[EXTERNAL DOWNLOAD LINK - Google Drive / Hugging Face / OneDrive / Kaggle]`
- **Telugu Model Checkpoint (`Telugu_FastSpeech2_latest.pth.tar`)**:
  `[EXTERNAL DOWNLOAD LINK - Google Drive / Hugging Face / OneDrive / Kaggle]`

Ensure the download links remain publicly accessible during the evaluation period.

---

## Documentation & Audits

- [`CHANGES.md`](CHANGES.md): Complete log of every file modified or added to the base FastSpeech 2 codebase, including justifications.
- [`DECISIONS.md`](DECISIONS.md): Architectural decisions, linguistic trade-offs, and rejected engineering alternatives.
- [`test_sentences.py`](test_sentences.py): Detailed byte-level Unicode anomaly audit and pronunciation challenges for test sentences.
- [`Report/Technical_Report.md`](Report/Technical_Report.md): Full technical report skeleton with factual sections completed.

