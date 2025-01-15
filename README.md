# Mask Out: A Novel Regularization Method for Code-Switching Language Identification

Mask Out is a novel method for word-level code-switching language identification, leveraging Multilingual BERT (mBERT) and a token-masking regularization technique. This approach enhances generalization in diverse linguistic contexts by selectively obscuring language-specific cues during training.

**Key Features**:
- Designed to improve code-switching detection across multilingual datasets.
- Utilizes a token-masking regularization method inspired by Dropout and MaskLID frameworks.
- Achieves optimal performance with a masking probability of 0.1.

## Dataset

The project uses datasets from the [LinCE benchmark](https://ritual.uh.edu/lince/datasets), focusing on:
- **SPA-ENG (Spanish-English)** for training.
- **NEP-ENG (Nepali-English)** for cross-dataset evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sean652039/code_switching_LID.git

2. Install dependencies:
    ```bach
    pip install -r requirements.txt

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Sean652039/code_switching_LID.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Train the model:
```bash
python train.py --dataset_path /path/to/dataset --mask_probability 0.1
```
4. Evaluate the model:
```bash
python evaluate.py --dataset_path /path/to/eval_dataset
```

## Acknowledgments

This work is inspired by prior research in multilingual NLP, including contributions by:
- **Dropout** by Nitish Srivastava et al.
- **MaskLID** by Amir Hossein Kargaran et al.
- **LinCE Benchmark** by Aguilar et al.