# Agent Evaluation — README

> Reproducible notebooks for (i) generating solutions + tests and (ii) evaluating final explanations.

---

## Repository layout (for reference)

    .
    ├── pipeline/
    │   └── Pipeline.ipynb                 # main pipeline (generate code, run tests)
    ├── evaluation/
    │   └── final_explanation_evaluation.ipynb  # evaluate explanations & export metrics
    ├── data/                               # sample data and example outputs (for reference)
    │   ├── LeetCodeDataset-v0.1.0-test.jsonl
    │   ├── Easy_version1_results.pkl
    │   ├── Medium_version1_results.pkl
    │   ├── Hard_version1_results.pkl
    │   └── evaluation_results.csv
    ├── requirements.txt
    └── environment.yaml

> **Note:** These files in `data/` are provided for reference and reproducibility checks.  
> When running the notebooks in Google Colab, please mount your own Google Drive and  
> update file paths accordingly (both for inputs and outputs).

---

## Prerequisites

- Python 3.10 recommended  
- Internet access (for first-time model downloads and `pistonpy` remote execution)  
- **Strongly recommended:** run `Pipeline.ipynb` on **Google Colab** with **A100 GPU**  
  - This matches our original setup and ensures smooth inference with StarCoder2-7B

---

## Setup
> **Note:** Google Colab comes with many standard libraries pre-installed  
> (e.g., `numpy`, `pandas`, `matplotlib`, `PyTorch`, `notebook`, etc.).  
> Therefore, `requirements.txt` and `environment.yaml` list only the  
> specialized dependencies that may **not** be available by default.  
> If you are running locally, make sure to install **all** listed dependencies  
> to match the environment used in our experiments.
> 
### Option A — conda (local, reproducible)

    conda env create -f environment.yaml
    conda activate agent-eval
    python -m spacy download en_core_web_sm

### Option B — pip (local)

    python -m venv .venv
    source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

> Hugging Face models (e.g., bigcode/starcoder2-7b, microsoft/codebert-base) will auto-download on first use.

---

## Data

For local runs, adjust dataset and output paths as needed.  
For Colab runs:

1. Mount Google Drive:
   
       from google.colab import drive
       drive.mount('/content/drive')

2. Update notebook paths, for example:

       dataset_path = "/content/drive/MyDrive/your_folder/LeetCodeDataset-v0.1.0-test.jsonl"
       output_path  = "/content/drive/MyDrive/your_folder/Easy_version1_results.pkl"

---

## Usage

### 1) Generate results (`pipeline/Pipeline.ipynb`)

- Recommended: Google Colab with **A100 GPU** (Runtime → Change runtime type → GPU = A100)  
- Mount Google Drive and set dataset/output paths to your Drive locations  
- Run all cells  
- Expected outputs: three `.pkl` result files saved to your specified Drive folder

> Notes: `pistonpy` runs tests in a remote sandbox; occasional network issues may require a retry.

### 2) Evaluate explanations (`evaluation/final_explanation_evaluation.ipynb`)

- Mount Google Drive and set `.pkl` file paths accordingly  
- Run all cells  
- Expected output: `.csv` file with evaluation metrics saved to your Drive

---

## Reproducibility tips

- Python 3.10, see `environment.yaml`/`requirements.txt`  
- For Colab runs, GPU A100 is strongly advised for `Pipeline.ipynb`  
- First run will download models (cache in `~/.cache/huggingface` or in Colab’s `/root/.cache/`)

---

## Troubleshooting

- **Out of memory (GPU)**: if not using A100, reduce batch size or switch to A100  
- **`pistonpy` errors**: transient; re-run cell or try again later  
- **spaCy errors**: ensure `en_core_web_sm` is installed: `python -m spacy download en_core_web_sm`

---

## requirements.txt

    # Core
    torch>=2.2.0
    transformers>=4.41.0
    tokenizers>=0.15.2
    datasets>=2.19.0
    pandas>=2.0.0
    numpy>=1.24.0
    scikit-learn>=1.3.0
    tqdm>=4.66.0
    ipykernel>=6.29.0
    jupyterlab>=4.1.0

    # Testing sandbox
    pistonpy>=1.7.0

    # Evaluation (explanations)
    textstat>=0.7.3
    spacy>=3.7.2
    textdescriptives>=2.10.0
    matplotlib>=3.7.0

> After installation, run: `python -m spacy download en_core_web_sm`

---

## environment.yaml

    name: agent-eval
    channels:
      - nvidia
      - pytorch
      - conda-forge
      - defaults
    dependencies:
      - python=3.10
      - pip
      # GPU stack (A100-friendly; remove if CPU only)
      - pytorch=2.2.*
      - pytorch-cuda=12.1
      - cudatoolkit
      # Scientific stack
      - numpy>=1.24
      - pandas>=2.0
      - scikit-learn>=1.3
      - matplotlib>=3.7
      - jupyterlab>=4.1
      - ipykernel>=6.29
      - spacy>=3.7
      - pip:
          - transformers>=4.41.0
          - tokenizers>=0.15.2
          - datasets>=2.19.0
          - pistonpy>=1.7.0
          - textstat>=0.7.3
          - textdescriptives>=2.10.0
          - tqdm>=4.66.0

> If you’re on CPU only, remove `pytorch-cuda` and `cudatoolkit`, and install the CPU build of PyTorch from the `pytorch` channel.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
