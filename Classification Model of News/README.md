````md
# News Classification using TF-IDF and LinearSVC

A multi-class news headline classification project that classifies news titles into six categories:

- Business
- Energy
- Health
- Markets
- Politics
- Technology

## Overview

The original dataset contained manually labeled news headlines, but many labels were inconsistent or conflicting.

To improve the dataset quality, a label-cleaning pipeline was applied before training the final classifier.

The final classification model uses:

**Word TF-IDF + Character TF-IDF + LinearSVC**

## Dataset

Input file:

```text
news.csv
````

Dataset statistics:

```text
Original Rows      : 5000
Valid Rows         : 4997
Unique Headlines   : 4497
Conflicting Titles : 123
```

Main columns:

```text
URL
Title
Category
Group Leader
```

## Pipeline

```text
news.csv
   ↓
Data Cleaning
   ↓
Duplicate / Conflict Handling
   ↓
Label Correction
   ↓
Corrected Dataset
   ↓
Train / Test Split
   ↓
Word + Character TF-IDF
   ↓
LinearSVC
   ↓
Predicted Category
```

## Label Correction

The original dataset contained inconsistent labels.

Labels were corrected in two steps:

1. Clear headlines were labeled using title-based rules and URL category information.
2. Unresolved headlines were labeled using a Logistic Regression model trained on trusted labels.

Logistic Regression is used only for label correction.

The final prediction model is **LinearSVC**.

## Final Corrected Distribution

```text
Markets       2044
Business       828
Technology     663
Politics       619
Health         217
Energy         126
```

## Feature Extraction

Two TF-IDF representations are used:

### Word TF-IDF

Uses word unigrams, bigrams, and trigrams.

```python
ngram_range=(1, 3)
```

### Character TF-IDF

Uses character n-grams.

```python
analyzer="char_wb"
ngram_range=(3, 5)
```

Both feature sets are combined using `FeatureUnion`.

## Final Model

```python
LinearSVC(
    C=0.10,
    class_weight="balanced",
    random_state=42
)
```

The model receives only the **news title** as input.

```text
News Title
   ↓
TF-IDF Features
   ↓
LinearSVC
   ↓
Category
```

## Train/Test Split

The dataset is split using unique normalized headlines to prevent the same headline from appearing in both training and testing.

```text
Training Rows         : 4005
Testing Rows          : 992
Unique Test Headlines : 900
```

## Results

```text
Accuracy          : 86.67%
Balanced Accuracy : 83.27%
Macro F1          : 83.41%
Weighted F1       : 86.70%
```

### Class-wise F1 Scores

| Category   | F1 Score |
| ---------- | -------: |
| Business   |   81.87% |
| Energy     |   72.34% |
| Health     |   83.72% |
| Markets    |   90.16% |
| Politics   |   85.71% |
| Technology |   86.67% |

## Project Structure

```text
News-Classification/
│
├── news.csv
├── FINAL_targeted_label_fix.py
├── corrected_news.csv
├── news_classifier.pkl
└── README.md
```

## Installation

```bash
pip install pandas numpy scikit-learn joblib
```

## Run

```bash
python FINAL_targeted_label_fix.py
```

After execution, the following files are generated:

```text
corrected_news.csv
news_classifier.pkl
```

## Prediction Example

```python
import joblib

model = joblib.load("news_classifier.pkl")

headline = ["Nvidia shares rise after strong earnings"]

prediction = model.predict(headline)

print(prediction[0])
```

Example output:

```text
Markets
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF
* Logistic Regression
* LinearSVC
* Joblib
* Regular Expressions

## Note

The original dataset contained noisy and inconsistent manual labels.

The test labels used in the final experiment were automatically corrected labels and were not fully human-verified.

## Future Work

* BiLSTM-based news classification
* Trainable word embeddings
* GloVe / FastText embeddings
* BERT-based classification
* Human-verified test dataset
* Improved handling of minority classes

```
```
