# News Headline Classification Using Deep Learning

A Deep Learning based **News Headline Classification** project that compares multiple neural network architectures using **PyTorch** and **pretrained GloVe word embeddings**.

The goal of this project is to classify a news headline into one of six categories and compare different deep learning models under the same experimental conditions.

## News Categories

The model predicts one of the following categories:

- Business
- Energy
- Health
- Markets
- Politics
- Technology

---

## Project Pipeline

```text
News Headline
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Vocabulary Creation
      ↓
Tokens → IDs
      ↓
Dynamic Padding
      ↓
Pretrained GloVe Embeddings
      ↓
ANN / RNN / LSTM / BiLSTM / GRU
      ↓
Dropout
      ↓
Linear Layer
      ↓
News Category
```

---

## Models Compared

Five neural network architectures were implemented and evaluated:

1. Artificial Neural Network (ANN)
2. Recurrent Neural Network (RNN)
3. Long Short-Term Memory (LSTM)
4. Bidirectional LSTM (BiLSTM)
5. Gated Recurrent Unit (GRU)

All models use the same:

- Dataset
- Train/Validation/Test split
- Text preprocessing
- Vocabulary
- GloVe embeddings
- Batch size
- Loss function
- Optimizer
- Evaluation metrics

This provides a fair comparison between the architectures.

---

## Dataset

The project uses a news headline dataset containing approximately **5,000 records**.

Important columns include:

```text
Title
Final_Category
```

After removing normalized duplicate headlines, approximately **4,497 unique headlines** remain.

The dataset is divided into:

```text
80% Training
10% Validation
10% Testing
```

Stratified splitting is used to preserve the class distribution.

---

## Text Preprocessing

Each news headline is processed using the following steps:

- Convert text to lowercase
- Remove punctuation and special characters
- Remove unnecessary spaces
- Tokenize the headline into words
- Remove empty token sequences
- Remove normalized duplicate headlines
- Convert tokens into numerical IDs

Example:

```text
Original:
"Nvidia's shares rise after strong earnings!"

Tokens:
["nvidia", "shares", "rise", "after", "strong", "earnings"]
```

---

## Vocabulary

The vocabulary is created using only the training dataset.

Special tokens:

```text
<PAD> = 0
<UNK> = 1
```

Minimum word frequency:

```python
MIN_FREQ = 2
```

Words not present in the vocabulary are mapped to `<UNK>`.

---

## Pretrained GloVe Embeddings

The project uses:

```text
glove-wiki-gigaword-100
```

Each word is represented using a **100-dimensional pretrained vector**.

GloVe helps the models use semantic information learned from a large text corpus instead of learning every word representation completely from scratch.

The embeddings are fine-tuned during model training.

```python
freeze=False
```

---

## Model Architectures

### ANN

```text
GloVe Embeddings
      ↓
Mean Pooling
      ↓
Dense Layer
      ↓
ReLU
      ↓
Dropout
      ↓
Output Layer
```

### RNN

```text
GloVe Embeddings
      ↓
2-Layer RNN
      ↓
Dropout
      ↓
Linear Layer
      ↓
Output
```

### LSTM

```text
GloVe Embeddings
      ↓
2-Layer LSTM
      ↓
Dropout
      ↓
Linear Layer
      ↓
Output
```

### BiLSTM

```text
GloVe Embeddings
      ↓
2-Layer Bidirectional LSTM
      ↓
Forward Hidden + Backward Hidden
      ↓
Concatenation
      ↓
Dropout
      ↓
Linear Layer
      ↓
Output
```

### GRU

```text
GloVe Embeddings
      ↓
2-Layer GRU
      ↓
Dropout
      ↓
Linear Layer
      ↓
Output
```

---

## Training Configuration

The main training configuration is:

```text
Embedding Dimension : 100
Hidden Dimension    : 128
Recurrent Layers    : 2
Dropout             : 0.3
Batch Size          : 32
Maximum Epochs      : 20
Early Stop Patience : 4
Learning Rate       : 0.0005
Weight Decay        : 0.0001
```

Optimizer:

```python
AdamW
```

Loss function:

```python
CrossEntropyLoss(label_smoothing=0.05)
```

Gradient clipping is also applied:

```python
clip_grad_norm_(model.parameters(), 1.0)
```

---

## Early Stopping

The models are monitored using **Validation Macro F1 Score**.

If validation performance does not improve for four consecutive epochs, training is stopped.

The checkpoint with the highest Validation Macro F1 is selected.

---

## Evaluation Metrics

The models are compared using:

- Test Accuracy
- Macro F1 Score
- Weighted F1 Score
- Validation Macro F1 Score

Macro F1 is especially important because the dataset is class-imbalanced.

---

## Final Model Comparison

| Model | Best Epoch | Validation Macro F1 | Test Accuracy | Test Macro F1 | Weighted F1 |
|------|------:|------:|------:|------:|------:|
| **ANN** | 20 | **78.96%** | **80.40%** | **72.43%** | **80.15%** |
| BiLSTM | 10 | 78.13% | 76.39% | 70.14% | 76.26% |
| GRU | 8 | 75.83% | 75.95% | 67.68% | 75.96% |
| LSTM | 14 | 71.40% | 75.28% | 66.80% | 75.15% |
| RNN | 11 | 72.86% | 74.83% | 65.55% | 74.69% |

---

## Best Model

The **Artificial Neural Network (ANN)** achieved the highest performance:

```text
Test Accuracy : 80.40%
Macro F1      : 72.43%
Weighted F1   : 80.15%
```

Therefore, ANN was selected as the best-performing architecture for this dataset.

An important observation from the experiment is that a more complex sequential model does not always guarantee better performance.

Since news headlines are relatively short, pretrained GloVe embeddings combined with mean pooling allowed the ANN to capture enough semantic information to perform strongly.

---

## Project Files

```text
News-Headline-Classification/
│
├── News_Classifier.ipynb
├── news.csv
├── model_comparison.csv
└── README.md
```

### `News_Classifier.ipynb`

Contains the complete step-by-step implementation including:

- Data loading
- Cleaning
- Tokenization
- Vocabulary creation
- GloVe embeddings
- Dynamic padding
- ANN
- RNN
- LSTM
- BiLSTM
- GRU
- Training
- Evaluation
- Final model comparison

### `news.csv`

Contains the news headline dataset used for the project.

### `model_comparison.csv`

Contains the final performance comparison of all five models.

---

## Installation

Install the required Python libraries:

```bash
pip install torch pandas numpy scikit-learn gensim
```

---

## Run in Google Colab

The project was developed as a Google Colab notebook.

Open:

```text
News_Classifier.ipynb
```

and run the cells sequentially.

If the dataset is stored in Google Drive, mount Drive using:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Then set the correct dataset path.

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Gensim
- GloVe Word Embeddings
- Google Colab

---

## Key Learning Outcomes

This project demonstrates:

- Natural Language Processing preprocessing
- Tokenization
- Vocabulary creation
- Word-to-ID conversion
- Dynamic padding
- Pretrained word embeddings
- ANN architecture
- Recurrent Neural Networks
- LSTM
- Bidirectional LSTM
- GRU
- Early stopping
- Gradient clipping
- Multiclass classification
- Model comparison and evaluation

---

## Conclusion

This project compared ANN, RNN, LSTM, BiLSTM, and GRU for multiclass news headline classification using the same preprocessing and training pipeline.

Among all tested architectures, **ANN achieved the highest test accuracy of 80.40%**.

The experiment demonstrates that the best architecture depends on the characteristics of the dataset and that more complex recurrent architectures are not necessarily superior for short-text classification tasks.

---

## Author

**Abdul Hadi**

