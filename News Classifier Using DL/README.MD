# News Headline Classification Using BiLSTM

A Deep Learning based news headline classification system built using **PyTorch**, **Pretrained GloVe Embeddings**, and a **2-Layer Bidirectional LSTM (BiLSTM)**.

The model classifies news headlines into six categories:

- Business
- Energy
- Health
- Markets
- Politics
- Technology

---

## Project Pipeline

```text
News Title
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
2-Layer Bidirectional LSTM
   ↓
Dropout
   ↓
Linear Layer
   ↓
News Category