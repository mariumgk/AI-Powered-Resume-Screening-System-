# AI-Powered Resume Intelligence & Decision Support System

This project implements an end-to-end, reproducible Resume Intelligence system:

- Semantic understanding via **MPNet embeddings** (`sentence-transformers/all-mpnet-base-v2`)
- Supervised **domain classification** with **Logistic Regression** on embeddings
- Unsupervised **clustering** with **K-Means** and **silhouette score**
- Mandatory **Resume ↔ Job Description matching** using **cosine similarity** + **skill overlap/gap analysis**
- Knowledge-based reasoning and constraints (rules only; no black-box LLM decisions)
- Offline training/evaluation, online inference via Flask web app

## Project structure

The repository follows the required structure under `resume-intelligence-system/`.

## Setup

Python version (recommended for reproducibility):

- Python **3.11 (64-bit)**

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Windows (PowerShell) recommendation:

```powershell
. ..\..\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt --upgrade
```

If activation is unreliable in your IDE terminal, you can always use the venv interpreter directly:

```powershell
..\..\.venv\Scripts\python.exe -m pip install -r requirements.txt --upgrade
```

To confirm your interpreter version:

```powershell
python -V
python -c "import sys; print(sys.executable)"
```

If you run the install command from the parent folder (`AI-Resume-System/`), use:

```bash
pip install -r resume-intelligence-system/requirements.txt
```

## Dataset

Place the screening-oriented dataset CSV at:

- `data/raw/updated_resume_dataset.csv`

Expected columns:

- `Resume` (resume text)
- `Category` (job domain label)

The training pipeline normalizes this internally to `id`, `resume_text`, and `label`.

## Offline training & evaluation (required)

From the project root (`resume-intelligence-system/`):

```bash
python -m src.main_pipeline
```

Artifacts are saved to `data/processed/artifacts/`.

## Screening-oriented framing

This project is designed for **resume screening and shortlisting**, not academic “one true label” classification.

- The classifier provides a **probability distribution over domains** (decision support), not a hard decision.
- The system uses **ranking signals** to support screening, including:
  - **Top-3 domains** (recommended domain shortlist)
  - **Top-3 accuracy** and **MRR** during evaluation (ranking quality)
  - **Resume ↔ Job Description cosine similarity** (semantic match)

Important note:

- The updated dataset is cleaner than real-world resumes; higher accuracy is expected. Real resumes are evaluated via inference behavior (screening + explanations), not by direct comparison of raw accuracy numbers.

## Windows note (PyTorch DLL / c10.dll)

`sentence-transformers` depends on `torch`. On Windows, you may see:

- `OSError: [WinError 1114] ... Error loading ... torch\lib\c10.dll`

This is an environment issue (not a project code issue). Common fixes:

1. Install/repair **Microsoft Visual C++ Redistributable 2015-2022 (x64)**.
2. Reinstall a CPU-only PyTorch build:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Troubleshooting: sentence-transformers + huggingface_hub

If you see:

- `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`

it means your installed `huggingface-hub` is newer than what `sentence-transformers==2.2.2` expects.

Fix:

```bash
pip install -r requirements.txt --upgrade
```

## Windows note: scikit-learn build tools error

If `pip` tries to install `scikit-learn` from source (`.tar.gz`) and you get:

- `Microsoft Visual C++ 14.0 or greater is required`

you are not getting a prebuilt wheel. Recommended fix (wheel-only install):

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: scikit-learn==1.5.2
python -m pip install -r requirements.txt
```

## Run the Flask app (inference only)

The web app **does not retrain** models. It only loads saved artifacts.

```bash
python -m src.app.flask_app
```

Then open:

- `http://127.0.0.1:5000`

## Viva justification notes (high-level)

- **Why MPNet embeddings?** Strong semantic similarity performance; supports matching, clustering, and classification in a shared embedding space.
- **Why minimal preprocessing?** Heavy NLP pipelines (lemmatization/stopwords) can distort semantics for transformer encoders; we only remove high-noise/PII patterns and normalize formatting.
- **Why Logistic Regression?** Probabilistic, class-balanced, explainable baseline on embeddings.
- **Why section-aware extraction?** Concentrates signal (skills/experience/summary) and reduces irrelevant text for embeddings.
- **Why rule-based reasoning?** Provides defensible, inspectable constraints and conflict resolution between ML outputs and knowledge graph evidence.

