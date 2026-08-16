News Title Clustering

Overview

This project applies unsupervised machine learning to group news titles into meaningful clusters using TF-IDF text features and engineered numerical features.

The clustering process does not use the original category labels during training. The original True_Category is preserved only for external evaluation after clustering.

Project Pipeline

Raw Dataset
→ Data Understanding
→ Data Cleaning
→ Feature Engineering
→ Candidate Features
→ Feature Selection
→ Feature Scaling
→ Dimensionality Reduction
→ Exploratory Data Analysis
→ Cluster Number Selection
→ Candidate Clustering Models
→ Internal Evaluation
→ External Evaluation
→ Cluster Profiling
→ Manual Cluster Naming
→ Outlier / Noise Detection
→ Deployment

Dataset

Original records: 4,997

Missing titles: 0

Duplicate titles: 423

Records after cleaning and duplicate removal: 4,449

Only the news title is used to create clustering features.

Data Cleaning

The titles are cleaned by:

Converting text to lowercase

Removing URLs

Removing punctuation

Removing extra spaces

Removing empty titles

Removing duplicate cleaned titles

Feature Engineering

The following numerical features are extracted:

word_count

char_count

avg_word_length

unique_word_ratio

number_count

year_present

trending_keyword_count

poi_count

company_keyword_count

TF-IDF Features

Text is converted into numerical vectors using TfidfVectorizer.

Main settings:

English stop-word removal

ngram_range=(1, 2)

min_df=2

max_features=3000

Final TF-IDF shape:

(4449, 3000)

Feature Scaling and Dimensionality Reduction

Numerical features are scaled using StandardScaler.

TF-IDF and numerical features are combined and reduced to 50 dimensions using TruncatedSVD.

Final reduced feature shape:

(4449, 50)

Clustering Models

Four clustering algorithms are compared:

KMeans

Agglomerative Clustering

Gaussian Mixture Model

DBSCAN

The final number of clusters used for the main models is:

K = 6

Internal Evaluation

Model

Clusters

Silhouette

Davies-Bouldin

Calinski-Harabasz

Noise

KMeans

6

0.281452

1.288999

853.660485

0

Agglomerative

6

0.271681

1.410509

813.780445

0

Gaussian Mixture

6

0.068793

3.065139

294.669898

0

DBSCAN

33

0.011375

0.867909

125.910444

1754

KMeans is selected as the final model because it provides the strongest overall internal clustering quality among the six-cluster candidate models, produces interpretable clusters, and supports prediction for unseen titles.

External Evaluation

The original category labels are used only after clustering for evaluation.

Metrics:

Clustering Accuracy

Adjusted Rand Index (ARI)

Normalized Mutual Information (NMI)

Model

Clustering Accuracy

ARI

NMI

KMeans

30.97%

0.010764

0.056402

Agglomerative

30.55%

0.014509

0.056704

Gaussian Mixture

36.66%

0.028632

0.100088

DBSCAN

27.98%

-0.003339

0.065463

Although Gaussian Mixture achieves higher external clustering accuracy, KMeans is selected because its internal clustering quality and interpretability are substantially better.

Clustering accuracy is not the primary metric because this is an unsupervised learning task.

Final KMeans Cluster Names

After cluster profiling, the following names are assigned:

kmeans_cluster_names = {
    0: "Business",
    1: "Markets",
    2: "Technology",
    3: "Corporate Announcements & Finance",
    4: "Earnings & Financial Results",
    5: "Politics"
}

The original category names are used where they naturally match the discovered cluster. Descriptive names are kept where forcing an original category would not accurately represent the cluster.

Cluster Profiling

Clusters are interpreted using:

Cluster size

Top TF-IDF terms

Trending keyword average

Point-of-interest keyword average

Company keyword average

Sample news titles

Noise Detection

DBSCAN is used to identify outliers and noise.

DBSCAN Noise Points: 1754

KMeans assigns every title to one of its six clusters.

Deployment

The final deployment pipeline is:

New News Title
→ Cleaning
→ Engineered Numerical Features
→ TF-IDF Transformation
→ Feature Scaling
→ Feature Combination
→ Truncated SVD
→ KMeans Prediction
→ Human-Readable Cluster Name

Example:

Title: Nvidia launches new AI chip for datacenters
Predicted Cluster: 2
Category: Technology

Output Files

The project generates:

clustered_news.csv

clustering_model_comparison.csv

clustering_external_evaluation.csv

Technologies Used

Python

Pandas

NumPy

Matplotlib

SciPy

Scikit-learn

Jupyter Notebook

Conclusion

KMeans is selected as the final clustering model with 6 clusters.

Final KMeans results:

Silhouette Score: 0.281452

Davies-Bouldin Score: 1.288999

Calinski-Harabasz Score: 853.660485

External Clustering Accuracy: 30.97%

ARI: 0.010764

NMI: 0.056402

The relatively low external agreement is expected because clustering is unsupervised and the naturally discovered groups do not necessarily match the predefined dataset categories exactly.