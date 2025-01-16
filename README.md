# Mask Out: A Novel Regularization Method for Code-Switching Language Identification

Mask Out is a novel method for word-level code-switching language identification, leveraging Multilingual BERT (mBERT) and a token-masking regularization technique. This approach enhances generalization in diverse linguistic contexts by selectively obscuring language-specific cues during training.

**Key Features**:
- Designed to improve code-switching detection across multilingual datasets
- Utilizes a token-masking regularization method inspired by Dropout and MaskLID frameworks
- Achieves optimal performance with a masking probability of 0.1

## Dataset

The project uses datasets from the [LinCE benchmark](https://ritual.uh.edu/lince/datasets), focusing on:
- **SPA-ENG (Spanish-English)** for training.
- **NEP-ENG (Nepali-English)** for cross-dataset evaluation.

## Usage

**1. Clone the repository**:
```bash
git clone https://github.com/Sean652039/code_switching_LID.git
```
**2. Install Dependencies**:
```bash
pip install -r requirements.txt
```
**3. Train the Model**:
```bash
python train.py
```
**4. Analyze Performance**: Generate performance plots based on training logs
```bash
python performance_vs_prob_plot.py
```
**5. Dataset Analysis**: Visualize dataset statistics
```bash
python analysis.py
```

## Technologies Used

- **Python**: The primary programming language used for all scripts
- **PyTorch**: For building and fine-tuning the Multilingual BERT model
- **Transformers**: To utilize pre-trained transformer models
- **Scikit-learn**: For evaluation metrics like F1-score, precision, and recall
- **Matplotlib**: For creating visualizations such as label distributions and performance plots
- **NumPy**: For numerical operations and dataset preprocessing
- **Tqdm**: To display progress bars during training and evaluation

## Acknowledgments

This work is inspired by prior research in multilingual NLP, including contributions by:
- **Dropout** by Nitish Srivastava et al.
- **MaskLID** by Amir Hossein Kargaran et al.
- **LinCE Benchmark** by Aguilar et al.