# Named Entity Recognition (NER) using Hugging Face

This project applies **Named Entity Recognition (NER)** to the `Title` column of a news dataset using a pretrained Hugging Face Transformer model.

## Objective

The goal is to identify named entities present in each news title and store them in a new `NER` column.

Each result is stored as a dictionary:

```python
{
    "Entity": "Entity Type"
}
```

Example:

```python
{
    "Elon Musk": "PERSON",
    "Tesla": "ORGANIZATION",
    "Germany": "PLACE"
}
```

## Model

**Model:** `dslim/bert-base-NER`

The model is based on **BERT** and is fine-tuned for Named Entity Recognition.

It detects entities such as:

* `PER` → PERSON
* `ORG` → ORGANIZATION
* `LOC` → PLACE
* `MISC` → MISCELLANEOUS

## Workflow

```text
News Title
    ↓
Hugging Face Tokenizer
    ↓
BERT NER Model
    ↓
Entity Detection
    ↓
Dictionary Conversion
    ↓
NER Column
```

## Example

Input:

```text
Elon Musk announced a new Tesla factory in Germany.
```

Output:

```python
{
    "Elon Musk": "PERSON",
    "Tesla": "ORGANIZATION",
    "Germany": "PLACE"
}
```

## Final Dataset

The original dataset is updated with a new `NER` column.

```text
Title | Category | NER
```

Example:

```text
Intel launches new processors | Technology | {"Intel": "ORGANIZATION"}
```

If no entity is detected, an empty dictionary is stored:

```python
{}
```

## Libraries

* Python
* Pandas
* PyTorch
* Hugging Face Transformers

## Output

The final dataset contains the original news data along with the extracted named entities in the `NER` column.
