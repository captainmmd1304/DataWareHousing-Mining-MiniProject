# Customer Segmentation with Hierarchical Clustering

A mini project comparing Agglomerative (bottom-up) and Divisive (top-down) clustering on customer data.

## Features
- Generated demo dataset of 500 customers
- Optional CSV upload
- Feature selection and cluster-count controls
- Agglomerative clustering with Ward linkage
- Divisive clustering using recursive two-way K-means splits
- Silhouette score and execution-time comparison
- Cluster scatter plots, dendrogram, cluster profiles, CSV export

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit (usually http://localhost:8501).

On some systems, use `python3 -m pip` and `python3 -m streamlit run app.py`.

## CSV format
Upload a CSV with at least two numeric features. Suggested columns:
- `Age`
- `AnnualIncome_k`
- `SpendingScore`
- `PurchasesPerMonth`

Any numeric columns can be selected in the sidebar. Missing rows in selected features are excluded from clustering.

## Algorithms
- **Agglomerative:** starts with one cluster per row and repeatedly merges nearest clusters.
- **Divisive:** starts with all rows in one cluster and repeatedly splits the largest cluster using K-means (`k=2`).

This divisive implementation is a practical educational heuristic; it is not a canonical implementation of every divisive hierarchical clustering algorithm.

## Suggested report title
**Comparative Analysis of Agglomerative and Divisive Hierarchical Clustering for Customer Segmentation**
