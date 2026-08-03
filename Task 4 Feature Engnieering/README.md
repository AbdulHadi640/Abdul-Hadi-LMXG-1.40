# News Article Feature Engineering

Pulls article titles from a Postgres database, tags each one with a set
of editorial-style features (category, trending topic, tone, etc.),
and writes the results to a new database table and a CSV file.

## Project structure

```
news_features_simple/
├── main.py                  # entry point — wires everything together
├── db.py                    # database connection, load, and save
├── feature_extraction.py   # all feature-tagging logic
├── .env                     # your database credentials (create this yourself)
└── articles_with_features.csv   # generated after running main.py
```

## Requirements

```bash
pip install pandas sqlalchemy psycopg2-binary python-dotenv
```

## Setup

1. Create a `.env` file in this folder with your database credentials:

   ```
   PG_DATABASE=news_scraper
   PG_HOST=localhost
   PG_PORT=5432
   PG_USER=postgres
   PG_PASSWORD=your_password_here
   ```

   Only `PG_PASSWORD` is required — the others default to the values
   shown above if omitted.

2. Make sure your database has an `articles` table with at least
   `id` and `title` columns.

## Running

```bash
python main.py
```

This will:
1. Connect to Postgres and load `id, title` from the `articles` table
2. Add a placeholder `clicks` column (random numbers — see note below)
3. Run every feature-tagging function on the titles
4. Print a sample of the results to the console
5. Write the full result to a new table called `article_features`
   (drops and recreates the table each run — see "Changing save
   behavior" below)
6. Also save a copy to `articles_with_features.csv` in this folder

## Features generated

| Column | Description |
|---|---|
| `word_count` | Number of words in the title |
| `category` | Primary news desk (Health, Politics, War & Conflict, Sports, etc.) |
| `subcategory` | Secondary desk if a second topic also applies, else `"None"` |
| `trending_topic` | The recurring word/story this title shares with many others in the dataset — computed from actual frequency, not a fixed list |
| `is_trending` | `Yes`/`No` — whether a trending topic was found |
| `tone` | `Positive` / `Negative` / `Neutral` |
| `severity` | `High` / `Moderate` / `Low` — how serious the story reads |
| `headline_style` | Breaking News / Explainer / Question / Analysis-Opinion / Recurring Feature / Straight News |
| `urgency` | `Urgent` / `Developing` / `Routine` |
| `story_scope` | `International` / `National` / `Local/Regional` |
| `story_type` | Obituary / Live Coverage / Recurring Column / Q&A-Explainer / News Report |
| `contains_quote` | `Yes`/`No` — title includes quotation marks |
| `contains_statistic` | `Yes`/`No` — title includes a number/percentage/dollar figure |
| `region` | Geographic region the story is about |
| `mentioned_entities` | Named people/organizations recognized in the title |
| `involves_children` | `Yes`/`No` |
| `audience` | Parents & Families / Policy Makers / Investors / Medical Professionals / General Public |

## Important notes

- **`clicks` is placeholder data.** It's filled with random numbers
  (`random.randint(5000, 10000)`) so the pipeline runs end to end.
  It has **no relationship to the actual title content** — don't
  train or evaluate any model against it until you swap in real
  click data.
- **`trending_topic` scales with dataset size.** The minimum number
  of repeats needed for a word to count as "trending" is calculated
  automatically (`max(3, 0.05% of total rows)`), so it behaves
  sensibly whether you're testing on 10 rows or running on 16,000.
  You can override it manually:
  ```python
  df = feature_engineering.compute_trending_topics(df, min_occurrences=10)
  ```

## Changing save behavior

By default, `db.save_features()` **replaces** the `article_features`
table every run. To append instead of overwrite, edit the call in
`main.py`:

```python
db.save_features(df_out, if_exists="append")
```

## Extending the feature set

All tagging logic lives in `feature_engineering.py`, organized by
section (category, trending, tone, style, region/entities, audience).
To add a new feature:

1. Write a function that takes a title string and returns a tag.
2. Add a line for it inside `build_features()`.
3. Add the new column name to `final_cols` in `main.py`.
