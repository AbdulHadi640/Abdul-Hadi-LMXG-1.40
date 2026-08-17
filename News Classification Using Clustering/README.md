# 📰 News Title Clustering

<p align="center">
  <b>Two-Stage Machine Learning: News Title Clustering with KMeans + Supervised KNN Prediction</b>
</p>

---

## 📌 Overview

This project uses a **two-stage machine learning workflow**.

### Stage 1 — Unsupervised Clustering
News titles are grouped into meaningful natural clusters using:

- **TF-IDF text features**
- **Engineered numerical features**
- **Feature scaling**
- **TruncatedSVD dimensionality reduction**
- **Multiple clustering algorithms**
- **Internal and external clustering evaluation**
- **Cluster profiling and manual human-readable naming**

### Stage 2 — Supervised Learning on Clustered Data
After KMeans creates the final cluster labels, those cluster IDs are treated as **pseudo-labels**.  
A **K-Nearest Neighbors (KNN)** classifier is then trained on the 50-dimensional SVD features to predict the discovered KMeans cluster for unseen news titles.

> **Important:** The original category labels are **not used during clustering** and are also **not used as the KNN target**.  
> `True_Category` is preserved only for **post-hoc external clustering evaluation**.  
> The KNN target is the **KMeans-generated cluster ID**.

---

## 🔄 Project Pipeline

```text
STAGE 1 — UNSUPERVISED CLUSTERING

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
TruncatedSVD (50 Dimensions)
    ↓
Exploratory Data Analysis
    ↓
Cluster Number Selection
    ↓
KMeans / Agglomerative / GMM / DBSCAN
    ↓
Internal Evaluation
    ↓
External Evaluation
    ↓
Best Clustering Model = KMeans
    ↓
KMeans Cluster Labels
    ↓
Cluster Profiling
    ↓
Manual Cluster Naming
    ↓
Outlier / Noise Analysis

STAGE 2 — SUPERVISED ML ON CLUSTERED DATA

X = 50-Dimensional SVD Features
y = KMeans Cluster IDs (Pseudo-Labels)
    ↓
80/20 Stratified Train/Test Split
    ↓
KNN
    ↓
Test K = 1, 3, 5, 7, 9
    ↓
Accuracy Comparison
    ↓
Best K = 5
    ↓
Best KNN Accuracy = 98.54%
    ↓
Classification Report
    ↓
Confusion Matrix
    ↓
Deployment on Unseen News Titles
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

## 🧠 Supervised ML on Clustered Data using KNN

After KMeans discovers the six natural clusters, the project moves to a second supervised stage.

The final KMeans cluster IDs:

```text
0, 1, 2, 3, 4, 5
```

are treated as **pseudo-labels**.

The supervised dataset is therefore:

```text
X = X_reduced
y = KMeans Cluster IDs
```

where `X_reduced` contains the **50-dimensional TruncatedSVD representation** of each news title.

> `True_Category` is **not** used as the KNN target.  
> KNN learns to reproduce the cluster structure discovered by KMeans.

### Train/Test Split

The clustered dataset is split using:

- **80% training data**
- **20% testing data**
- `random_state=42`
- `stratify=y_knn`

Stratification preserves approximately the same cluster proportions in both the training and test sets, which is important because the KMeans clusters have unequal sizes.

The test set contains:

```text
890 samples
```

### KNN Neighbour Selection

The following odd values of K are evaluated:

```text
K = 1, 3, 5, 7, 9
```

In **KNN**, K means the **number of nearest neighbours** used for voting.

This is different from **KMeans**, where K means the **number of clusters**.

### Best KNN Result

The best result is:

| Metric | Result |
|---|---:|
| Best K | **5** |
| Best KNN Accuracy | **98.54%** |
| Test Samples | **890** |
| Macro Precision | **0.99** |
| Macro Recall | **0.98** |
| Macro F1-score | **0.99** |
| Weighted Precision | **0.99** |
| Weighted Recall | **0.99** |
| Weighted F1-score | **0.99** |

### KNN Classification Report

| Cluster | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.99 | 1.00 | 0.99 | 426 |
| 1 | 0.96 | 1.00 | 0.98 | 113 |
| 2 | 1.00 | 1.00 | 1.00 | 36 |
| 3 | 0.98 | 0.91 | 0.95 | 128 |
| 4 | 1.00 | 1.00 | 1.00 | 81 |
| 5 | 1.00 | 1.00 | 1.00 | 106 |

Cluster **3** is the relatively most difficult cluster, with a recall of **0.91**, while clusters **2, 4, and 5** achieve perfect precision, recall, and F1-score on this test split.

### What Does 98.54% Accuracy Mean?

The **98.54% KNN accuracy is not original news-category classification accuracy**.

It means:

> KNN reproduces the **KMeans-generated cluster assignments** on unseen test samples with 98.54% accuracy.

This is different from the **30.97% KMeans external clustering accuracy**, which compares the discovered KMeans clusters with the original `True_Category` labels after optimal Hungarian matching.

Therefore:

```text
30.97% = KMeans clusters vs original dataset categories

98.54% = KNN predictions vs KMeans-generated pseudo-labels
```

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

The project uses **KMeans as the clustering model** and the **best KNN model (K=5) as the final supervised prediction stage**.

KMeans first discovers the cluster structure during training.  
KNN then learns those KMeans-generated cluster IDs and is used to predict the cluster of an unseen title.

### Prediction Pipeline

```text
New News Title
    ↓
Cleaning
    ↓
Engineered Numerical Features
    ↓
Existing TF-IDF Transformation
    ↓
Existing Feature Scaling
    ↓
Feature Combination
    ↓
Existing TruncatedSVD Transformation
    ↓
Best KNN Prediction (K = 5)
    ↓
Predicted KMeans Cluster ID
    ↓
Human-Readable Cluster Name
```

The fitted TF-IDF vectorizer, scaler, and SVD model are **reused with `.transform()`** during deployment. They are not fitted again on the new title.

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

### Stage 1 — KMeans Clustering

| Metric | Result |
|---|---:|
| Number of Clusters | **6** |
| Silhouette Score | **0.281452** |
| Davies-Bouldin Score | **1.288999** |
| Calinski-Harabasz Score | **853.660485** |
| External Clustering Accuracy | **30.97%** |
| ARI | **0.010764** |
| NMI | **0.056402** |

### Stage 2 — KNN on KMeans Pseudo-Labels

| Metric | Result |
|---|---:|
| K values tested | **1, 3, 5, 7, 9** |
| Best K | **5** |
| Best KNN Test Accuracy | **98.54%** |
| Test Samples | **890** |
| Macro F1-score | **0.99** |
| Weighted F1-score | **0.99** |

> The two accuracy values measure different things.  
> **30.97%** measures KMeans cluster agreement with the original dataset categories after Hungarian matching.  
> **98.54%** measures how accurately KNN predicts the KMeans-generated pseudo-labels.

---|---:|
| Number of Clusters | **6** |
| Silhouette Score | **0.281452** |
| Davies-Bouldin Score | **1.288999** |
| Calinski-Harabasz Score | **853.660485** |
| External Clustering Accuracy | **30.97%** |
| ARI | **0.010764** |
| NMI | **0.056402** |

---

## 📝 Conclusion

This project follows a **two-stage machine learning workflow**.

In the first stage, **KMeans** is selected as the final unsupervised clustering model with **6 clusters** because it provides the strongest overall internal clustering quality among the six-cluster candidate models. It achieves a Silhouette Score of **0.281452**, Davies-Bouldin Score of **1.288999**, and Calinski-Harabasz Score of **853.660485**.

The KMeans external clustering accuracy is approximately **30.97%**. This should not be interpreted like ordinary supervised classification accuracy because `True_Category` was never used to create the clusters. The naturally discovered groups are based on textual and engineered-feature similarity and therefore do not necessarily reproduce the predefined dataset categories.

In the second stage, the KMeans-generated cluster IDs are treated as **pseudo-labels** for a supervised **KNN classifier**. K values **1, 3, 5, 7, and 9** are compared. The best value is **K=5**, which achieves **98.54% test accuracy** on **890 held-out samples**. The classification report shows a macro F1-score of approximately **0.99** and a weighted F1-score of approximately **0.99**.

The KNN accuracy represents how accurately KNN reproduces the KMeans-generated cluster assignments. It does **not** represent accuracy against the original news categories.

Overall, the project demonstrates:

- Data understanding and cleaning
- TF-IDF text representation
- Numerical feature engineering
- Feature selection / limitation
- Feature scaling
- TruncatedSVD dimensionality reduction
- Multi-model clustering comparison
- Internal clustering evaluation
- External post-hoc evaluation
- Hungarian cluster-to-category matching
- Cluster profiling
- Manual human-readable naming
- DBSCAN noise detection
- Supervised learning on KMeans pseudo-labels
- KNN hyperparameter comparison
- Classification report and confusion matrix evaluation
- Deployment for unseen news titles

---

## ⭐ Key Takeaway

> **KMeans discovers the natural groups in the news titles, while KNN learns to predict those discovered groups for unseen titles.**

> **KMeans external clustering accuracy = 30.97%, while KNN pseudo-label prediction accuracy = 98.54%. These metrics answer different questions and should not be compared as if they were the same type of accuracy.**
