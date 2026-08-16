# 📰 News Title Clustering

<p align="center">
  <b>Unsupervised Machine Learning for Discovering Natural Topics in News Headlines</b>
</p>

---

## 📌 Overview

This project applies **unsupervised machine learning** to group news titles into meaningful clusters using a combination of:

- **TF-IDF text features**
- **Engineered numerical features**
- **Dimensionality reduction**
- **Multiple clustering algorithms**
- **Internal and external clustering evaluation**

> **Important:** The original category labels are **not used during clustering**.  
> `True_Category` is preserved only for **post-hoc external evaluation** after the clusters have already been created.

---

## 🔄 Project Pipeline

```text
Raw Dataset
    ↓
Data Understanding
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
TF-IDF Text Representation
    ↓
Candidate Features
    ↓
Feature Selection
    ↓
Feature Scaling
    ↓
Feature Combination
    ↓
Dimensionality Reduction
    ↓
Exploratory Data Analysis
    ↓
Cluster Number Selection
    ↓
Candidate Clustering Models
    ↓
Internal Evaluation
    ↓
External Evaluation
    ↓
Best Model Selection
    ↓
Cluster Profiling
    ↓
Manual Cluster Naming
    ↓
Outlier / Noise Detection
    ↓
Deployment
```

---

## 📊 Dataset

| Property | Value |
|---|---:|
| Original records | **4,997** |
| Missing titles | **0** |
| Duplicate titles | **423** |
| Records after cleaning | **4,449** |

Only the **news title** is used to create clustering features.

The original category column is renamed to `True_Category` and kept only for final external evaluation.

---

## 🧹 Data Cleaning

The news titles are cleaned by:

- Converting text to lowercase
- Removing URLs
- Removing punctuation
- Removing extra spaces
- Removing empty titles
- Removing duplicate cleaned titles

After cleaning, the final dataset contains **4,449 unique usable news titles**.

---

## 🛠️ Feature Engineering

Along with TF-IDF text features, the project extracts the following numerical features:

| Feature | Purpose |
|---|---|
| `word_count` | Number of words in the title |
| `char_count` | Number of characters |
| `avg_word_length` | Average word length |
| `unique_word_ratio` | Ratio of unique words to total words |
| `number_count` | Number of numerical values |
| `year_present` | Detects years such as 2025, 2026, etc. |
| `trending_keyword_count` | Counts keywords such as AI, stock, crypto, oil, earnings |
| `poi_count` | Counts location / geopolitical terms |
| `company_keyword_count` | Counts major company names such as Nvidia, Apple, Microsoft, OpenAI |

These engineered features provide additional structural and domain-specific information that complements TF-IDF.

---

## 🔤 TF-IDF Text Features

The cleaned titles are converted into numerical vectors using `TfidfVectorizer`.

### Main Settings

- English stop-word removal
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_features=3000`

### Why Unigrams + Bigrams?

Unigrams capture individual words such as:

- `stock`
- `market`
- `china`
- `nvidia`

Bigrams capture useful two-word phrases such as:

- `stock market`
- `china trade`
- `financial results`
- `quarter 2025`

### Final TF-IDF Shape

```text
(4449, 3000)
```

This means:

- **4,449 news titles**
- **3,000 TF-IDF text features**

---

## ⚖️ Feature Scaling

The engineered numerical features have different ranges.

For example:

- `year_present` is usually `0` or `1`
- `char_count` can be much larger
- `number_count` may contain small integer values

Therefore, the numerical features are standardized using **StandardScaler** so that one feature does not dominate the clustering process simply because of its scale.

The project uses:

```text
NUMERIC_WEIGHT = 1.0
```

The scaled numerical features are then combined with the TF-IDF matrix.

---

## 📉 Dimensionality Reduction

The combined feature matrix is high-dimensional, so **TruncatedSVD** is applied.

The feature space is reduced to:

```text
50 dimensions
```

Final reduced shape:

```text
(4449, 50)
```

### Why TruncatedSVD?

It helps:

- Reduce dimensionality
- Remove redundancy
- Speed up clustering
- Make distance-based clustering more manageable
- Work efficiently with sparse TF-IDF representations

---

## 📈 Exploratory Data Analysis

A word-count histogram is used to inspect the distribution of headline lengths.

This provides a quick understanding of the general structure of the news titles before clustering.

---

## 🔢 Selecting the Number of Clusters

Different values of `K` from **2 to 10** are inspected using:

### Elbow Method

The Elbow Method uses **inertia / WCSS**.

Lower inertia means points are closer to their assigned cluster centroids.

The goal is to identify the point where increasing `K` gives only a small additional improvement.

### Silhouette Score

Silhouette Score measures:

- How compact samples are within their own cluster
- How well separated they are from neighbouring clusters

Higher values indicate better cluster separation.

### Final Choice

```text
K = 6
```

`K=6` is used as a practical and interpretable six-cluster solution after inspecting the Elbow and Silhouette analyses.

---

## 🤖 Clustering Models

Four clustering algorithms are compared:

1. **KMeans**
2. **Agglomerative Clustering**
3. **Gaussian Mixture Model**
4. **DBSCAN**

### KMeans

A centroid-based clustering algorithm that assigns each sample to the nearest cluster centroid.

### Agglomerative Clustering

A hierarchical clustering algorithm that repeatedly merges the closest clusters.

### Gaussian Mixture Model

A probabilistic clustering approach that models the data as a mixture of Gaussian distributions.

### DBSCAN

A density-based clustering algorithm that can automatically identify dense regions and mark low-density samples as noise.

---

## 📏 Internal Evaluation

Because this is an **unsupervised clustering project**, internal metrics are the primary way to compare cluster quality.

### Metrics

- **Silhouette Score** → Higher is better
- **Davies-Bouldin Score** → Lower is better
- **Calinski-Harabasz Score** → Higher is better
- **Noise Points** → Mainly relevant for DBSCAN

### Results

| Model | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Noise |
|---|---:|---:|---:|---:|---:|
| **KMeans** | **6** | **0.281452** | **1.288999** | **853.660485** | **0** |
| Agglomerative | 6 | 0.271681 | 1.410509 | 813.780445 | 0 |
| Gaussian Mixture | 6 | 0.068793 | 3.065139 | 294.669898 | 0 |
| DBSCAN | 33 | 0.011375 | 0.867909 | 125.910444 | 1754 |

---

## 🏆 Best Model Selection

**KMeans** is selected as the final clustering model.

### Why KMeans?

KMeans provides the strongest overall internal clustering quality among the six-cluster candidate models.

It has:

- The highest Silhouette Score among the six-cluster models
- A better Davies-Bouldin Score than Agglomerative
- The highest Calinski-Harabasz Score
- Interpretable clusters
- No noise points
- Direct `.predict()` support for unseen titles

Although DBSCAN has a lower Davies-Bouldin value, it generates **33 clusters** and **1,754 noise points**, making it unsuitable for the required practical and interpretable six-cluster solution.

---

## 🧪 External Evaluation

The original category labels are used only **after clustering** to measure how closely the discovered clusters match the predefined categories.

### Metrics

- **Clustering Accuracy**
- **Adjusted Rand Index (ARI)**
- **Normalized Mutual Information (NMI)**

### Results

| Model | Clustering Accuracy | ARI | NMI |
|---|---:|---:|---:|
| KMeans | **30.97%** | 0.010764 | 0.056402 |
| Agglomerative | 30.55% | 0.014509 | 0.056704 |
| Gaussian Mixture | **36.66%** | **0.028632** | **0.100088** |
| DBSCAN | 27.98% | -0.003339 | 0.065463 |

---

## 🧩 Hungarian Cluster-to-Category Matching

Cluster IDs are arbitrary.

For example:

```text
Cluster 0
Cluster 1
Cluster 2
```

do not automatically correspond to:

```text
Business
Markets
Technology
```

Therefore, the **Hungarian algorithm** is used during external evaluation to find the optimal one-to-one mapping between discovered cluster IDs and original dataset categories.

This mapping is used only for **post-hoc evaluation** and does not affect how the clusters were created.

---

## ❓ Why KMeans Instead of Gaussian Mixture?

Gaussian Mixture achieves the highest external clustering accuracy:

```text
36.66%
```

However, clustering accuracy is **not the primary metric** in an unsupervised problem.

Gaussian Mixture has much weaker internal cluster quality:

- Silhouette: `0.068793`
- Davies-Bouldin: `3.065139`
- Calinski-Harabasz: `294.669898`

KMeans performs substantially better internally:

- Silhouette: `0.281452`
- Davies-Bouldin: `1.288999`
- Calinski-Harabasz: `853.660485`

Therefore, **KMeans is selected because it produces stronger natural cluster structure, better interpretability, and supports deployment for unseen titles**.

---

## 🏷️ Final KMeans Cluster Names

After cluster profiling, human-readable names are assigned:

```python
kmeans_cluster_names = {
    0: "Business",
    1: "Markets",
    2: "Technology",
    3: "Corporate Announcements & Finance",
    4: "Earnings & Financial Results",
    5: "Politics"
}
```

The original category names are used where they naturally match the discovered cluster.

Descriptive names are kept where forcing an original category would not accurately represent the natural cluster.

---

## 🔎 Cluster Profiling

Each cluster is interpreted using:

- Cluster size
- Top TF-IDF terms
- Trending keyword average
- Point-of-interest keyword average
- Company keyword average
- Sample news titles

### Cluster 0 — Business

**Size:** 2,128

Top terms include:

`new`, `says`, `announces`, `trump`, `shares`, `higher`, `million`, `global`, `group`, `profit`

This is a broad business and general corporate-news cluster.

---

### Cluster 1 — Markets

**Size:** 568

Top terms include:

`ai`, `stock`, `stocks`, `market`, `earnings`, `oil`, `crypto`, `inflation`, `bank`, `bitcoin`

Average trending keyword count:

```text
1.21
```

This cluster strongly represents stock markets, investments, crypto, earnings, and financial-market activity.

---

### Cluster 2 — Technology

**Size:** 179

Top terms include:

`nvidia`, `apple`, `ai`, `openai`, `meta`, `amazon`, `microsoft`, `data`, `pro`, `google`

Average company keyword count:

```text
1.15
```

This cluster clearly represents technology, AI, major technology companies, and related products.

---

### Cluster 3 — Corporate Announcements & Finance

**Size:** 638

Top terms include:

`million`, `group`, `billion`, `new`, `faruqi`, `shares`, `revenue`, `plc`, `announces`, `000`

This cluster contains many company announcements, financial amounts, revenue reports, shares, and corporate updates.

---

### Cluster 4 — Earnings & Financial Results

**Size:** 407

Top terms include:

`2025`, `quarter`, `quarter 2025`, `results`, `2026`, `earnings`, `october`, `financial results`, `financial`, `announces`

This cluster strongly represents quarterly results, earnings releases, dated financial reports, and company result announcements.

---

### Cluster 5 — Politics

**Size:** 529

Top terms include:

`china`, `india`, `says`, `trade`, `uk`, `israel`, `trump`, `china trade`, `tensions`, `billion`

Average POI count:

```text
1.19
```

This cluster represents politics, international affairs, trade tensions, countries, sanctions, and geopolitical news.

---

## 🚨 Noise / Outlier Detection

DBSCAN is used to identify low-density samples and possible outliers.

In DBSCAN:

```text
Cluster label -1 = Noise
```

Number of detected noise points:

```text
1754
```

KMeans itself does not create a noise category because every sample is assigned to one of its six clusters.

---

## 🚀 Deployment

The final deployment model is **KMeans** because it supports prediction for unseen titles.

### Prediction Pipeline

```text
New News Title
    ↓
Cleaning
    ↓
Engineered Numerical Features
    ↓
TF-IDF Transformation
    ↓
Feature Scaling
    ↓
Feature Combination
    ↓
TruncatedSVD Transformation
    ↓
KMeans Prediction
    ↓
Human-Readable Cluster Name
```

### Example

Input:

```text
Nvidia launches new AI chip for datacenters
```

Output:

```text
Predicted Cluster: 2
Category: Technology
```

---

## 📁 Output Files

The project generates the following files:

### `clustered_news.csv`

Contains:

- Original title
- Cleaned title
- True category
- KMeans cluster
- Assigned human-readable category

### `clustering_model_comparison.csv`

Contains the internal evaluation results for all candidate clustering models.

### `clustering_external_evaluation.csv`

Contains:

- Clustering Accuracy
- ARI
- NMI
- Evaluated Points
- Noise Excluded

---

## 🧰 Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **SciPy**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 📌 Final Results

| Metric | KMeans Result |
|---|---:|
| Number of Clusters | **6** |
| Silhouette Score | **0.281452** |
| Davies-Bouldin Score | **1.288999** |
| Calinski-Harabasz Score | **853.660485** |
| External Clustering Accuracy | **30.97%** |
| ARI | **0.010764** |
| NMI | **0.056402** |

---

## 📝 Conclusion

KMeans is selected as the final clustering model with **6 clusters** because it provides the strongest overall internal clustering quality among the six-cluster candidate models while also producing understandable clusters and supporting prediction for unseen news titles.

The external clustering accuracy is approximately **30.97%**. This value should not be interpreted like supervised classification accuracy because the original category labels were never used to create the clusters.

The naturally discovered groups are based on textual and engineered-feature similarity, so they do not necessarily match the predefined dataset categories exactly.

Overall, the project demonstrates a complete unsupervised text-clustering workflow including:

- Data cleaning
- TF-IDF representation
- Feature engineering
- Feature scaling
- Dimensionality reduction
- Multi-model comparison
- Internal and external evaluation
- Hungarian matching
- Cluster profiling
- Manual semantic naming
- Noise detection
- Deployment for unseen titles

---

## ⭐ Key Takeaway

> **The goal of this project is not to reproduce the original category labels. The goal is to discover meaningful natural groups in news titles using unsupervised learning.*
