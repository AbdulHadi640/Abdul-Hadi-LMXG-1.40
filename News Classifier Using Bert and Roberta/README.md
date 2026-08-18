# News Category Classification Using BERT and RoBERTa

This project performs **multi-class news category classification** using pretrained **BERT** and **RoBERTa** transformer models.

The models are fine-tuned on a news dataset and predict the category of a news headline.

## Categories

The dataset contains six news categories:

* Business
* Energy
* Health
* Markets
* Politics
* Technology

## Project Pipeline

```text
News Dataset
     ↓
Data Cleaning
     ↓
Label Encoding
     ↓
Train / Validation / Test Split
     ↓
Transformer Tokenization
     ↓
BERT / RoBERTa
     ↓
Fine-Tuning
     ↓
Evaluation
     ↓
News Category Prediction
```

## Dataset

The main columns used in the project are:

| Column           | Description                       |
| ---------------- | --------------------------------- |
| `Title`          | News headline used as model input |
| `Final_Category` | Target news category              |

The dataset is divided into:

* 80% Training Data
* 10% Validation Data
* 10% Test Data

Stratified splitting is used to preserve class distribution.

## Models

### BERT

Pretrained model:

```text
google-bert/bert-base-uncased
```

### RoBERTa

Pretrained model:

```text
FacebookAI/roberta-base
```

Both pretrained models are fine-tuned using a sequence classification head with six output classes.

## Installation

Install the required libraries:

```bash
pip install transformers datasets accelerate scikit-learn pandas numpy torch matplotlib
```

## Import Libraries

```python
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
```

## BERT Tokenization

```python
BERT_MODEL_NAME = "google-bert/bert-base-uncased"

bert_tokenizer = AutoTokenizer.from_pretrained(
    BERT_MODEL_NAME
)

def bert_tokenize(batch):
    return bert_tokenizer(
        batch["Title"],
        truncation=True,
        max_length=128
    )
```

## BERT Model

```python
bert_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME,
    num_labels=6,
    id2label=id2label,
    label2id=label2id
)
```

## RoBERTa Model

```python
ROBERTA_MODEL_NAME = "FacebookAI/roberta-base"

roberta_tokenizer = AutoTokenizer.from_pretrained(
    ROBERTA_MODEL_NAME
)

roberta_model = AutoModelForSequenceClassification.from_pretrained(
    ROBERTA_MODEL_NAME,
    num_labels=6,
    id2label=id2label,
    label2id=label2id
)
```

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Macro F1 Score
* Weighted F1 Score
* Precision
* Recall
* Per-class F1 Score

Macro F1 is especially useful because the dataset contains an unequal number of examples across categories.

## Results

| Model    |   Accuracy |   Macro F1 | Weighted F1 |
| -------- | ---------: | ---------: | ----------: |
| **BERT** | **77.73%** | **69.74%** |  **77.39%** |
| RoBERTa  |     73.80% |     65.65% |      74.01% |

BERT achieved the best overall performance on this dataset.

## BERT Per-Class Performance

| Category   | F1 Score |
| ---------- | -------: |
| Business   |   66.26% |
| Energy     |   50.00% |
| Health     |   65.00% |
| Markets    |   85.92% |
| Politics   |   75.20% |
| Technology |   76.06% |

The model performed best on the **Markets** category.

The **Energy** category was more difficult to classify because it contained comparatively fewer samples.

## Prediction Example

```python
def predict_with_bert(title):

    inputs = bert_tokenizer(
        title,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(bert_model.device)
        for key, value in inputs.items()
    }

    bert_model.eval()

    with torch.no_grad():
        outputs = bert_model(**inputs)

    predicted_id = torch.argmax(
        outputs.logits,
        dim=-1
    ).item()

    return id2label[predicted_id]
```

Example:

```python
news = "Microsoft announces a new artificial intelligence platform"

prediction = predict_with_bert(news)

print("Predicted Category:", prediction)
```

Possible output:

```text
Predicted Category: Technology
```

## Data Leakage and Overfitting

The dataset was split into training, validation, and test sets before model fine-tuning.

The training set was used for model training, while the test set remained unseen during training.

Exact duplicate headlines were removed before splitting to reduce the possibility of duplicate information appearing across different subsets.

A small difference between training and validation performance may indicate some overfitting tendency, but the final BERT model still generalized reasonably well on unseen test data.

## Conclusion

This project demonstrates how pretrained transformer models can be fine-tuned for news category classification.

Both BERT and RoBERTa were successfully trained and evaluated on the same dataset.

BERT achieved the strongest performance with:

* **77.73% Accuracy**
* **69.74% Macro F1**
* **77.39% Weighted F1**

Therefore, **BERT was selected as the best-performing model for this news classification task**.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Google Colab
