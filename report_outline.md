# Mini Project Report Outline

## Title
Comparative Analysis of Agglomerative and Divisive Hierarchical Clustering for Customer Segmentation

## Abstract
This project compares bottom-up agglomerative clustering with a top-down divisive clustering heuristic for discovering customer groups. Customer attributes are standardized before clustering. The resulting groups are visualized and compared using silhouette score and execution time.

## Problem Statement
Retailers have customers with different purchasing patterns. The goal is to group customers with similar characteristics without pre-existing labels and compare two hierarchical clustering strategies.

## Objectives
1. Prepare and standardize customer data.
2. Implement agglomerative clustering.
3. Implement divisive clustering using recursive two-way K-means splits.
4. Visualize clusters and an agglomerative dendrogram.
5. Compare silhouette scores, runtime, and cluster profiles.

## Methodology
1. Load a customer CSV or generate demo data.
2. Select numeric features and remove rows with missing selected features.
3. Standardize features using StandardScaler.
4. Fit both clustering methods for a selected number of clusters.
5. Evaluate using silhouette score and execution time.
6. Interpret average feature values in each cluster.

## Algorithms
### Agglomerative
Starts with each data point in its own cluster and repeatedly merges the closest clusters until the requested cluster count is reached.

### Divisive
Starts with all points in one cluster and repeatedly splits the largest cluster into two groups using K-means with k=2 until the requested count is reached.

## Evaluation
- Silhouette score: higher generally indicates better-separated clusters.
- Execution time: measures approximate runtime on the same data and machine.
- Cluster profiles: mean feature values help describe the groups.

## Conclusion
Write this after running the app. State which method had the higher silhouette score on your chosen data, compare runtime, and describe the discovered customer profiles. Results can change with the dataset, features, and cluster count.

## Limitations
The demo dataset is synthetic. Divisive clustering is implemented as a simple heuristic, and a higher silhouette score does not automatically imply business usefulness.
