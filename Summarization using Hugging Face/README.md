# News Summarization using Hugging Face

This project applies **abstractive text summarization** to a grouped news dataset using the pretrained **`facebook/bart-large-cnn`** model from Hugging Face.

The dataset contains news articles grouped by source. Each article includes fields such as `title`, `url`, `published_date`, and `first_paragraph`.

## Objective

The goal is to generate a short summary for each news article by combining:

```text
Title + First Paragraph
```

and passing the combined text to the BART summarization model.

## Model

**Model:** `facebook/bart-large-cnn`

BART is a Transformer-based sequence-to-sequence model designed for text generation tasks such as summarization.

## Workflow

```text
JSON News Dataset
        ↓
Title + First Paragraph
        ↓
Tokenization
        ↓
BART Summarization Model
        ↓
Generated Summary
        ↓
Summary added to JSON
```

## Output

A new `summary` field is added to every article.

Example:

```json
{
  "title": "Japan Kumamoto earthquake: What happened and what we know so far",
  "first_paragraph": "A magnitude 7.1 earthquake struck Japan’s Kyushu island...",
  "summary": "A powerful earthquake struck Japan's Kyushu island, causing widespread damage."
}
```

## Libraries

* Python
* PyTorch
* Hugging Face Transformers
* JSON
* tqdm

## Output File

```text
scraped_articles_with_summaries.json
```
