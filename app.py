import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title="Hierarchical Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Customer Segmentation with Hierarchical Clustering")
st.caption("Comparative analysis of Agglomerative (bottom-up) and Divisive (top-down) clustering")

@st.cache_data
def make_demo_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    profiles = [
        (25, 35, 25, 2.0),
        (45, 75, 80, 6.0),
        (35, 25, 65, 3.5),
        (55, 55, 35, 4.0),
        (22, 65, 45, 1.5),
    ]
    rows = []
    for i in range(n):
        p = profiles[rng.integers(0, len(profiles))]
        age = np.clip(rng.normal(p[0], 6), 18, 75)
        income = np.clip(rng.normal(p[1], 12), 10, 150)
        spending = np.clip(rng.normal(p[2], 13), 1, 100)
        frequency = max(1, rng.poisson(p[3]) + 1)
        rows.append([f"C{i+1:04d}", round(age, 1), round(income, 1), round(spending, 1), int(frequency)])
    return pd.DataFrame(rows, columns=["CustomerID", "Age", "AnnualIncome_k", "SpendingScore", "PurchasesPerMonth"])

def divisive_clustering(X, n_clusters, random_state=42):
    """Simple top-down divisive clustering using recursive 2-means splits."""
    labels = np.zeros(X.shape[0], dtype=int)
    clusters = {0: np.arange(X.shape[0])}
    next_id = 1
    while len(clusters) < n_clusters:
        splittable = [(cid, idx) for cid, idx in clusters.items() if len(idx) >= 2]
        if not splittable:
            break
        # Split the largest cluster first.
        cid, idx = max(splittable, key=lambda item: len(item[1]))
        if len(idx) < 2:
            break
        km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
        local_labels = km.fit_predict(X[idx])
        left, right = idx[local_labels == 0], idx[local_labels == 1]
        if len(left) == 0 or len(right) == 0:
            break
        del clusters[cid]
        clusters[next_id] = left
        next_id += 1
        clusters[next_id] = right
        next_id += 1
    final_labels = np.zeros(X.shape[0], dtype=int)
    for new_label, idx in enumerate(clusters.values()):
        final_labels[idx] = new_label
    return final_labels

uploaded = st.sidebar.file_uploader("Upload your customer CSV (optional)", type=["csv"])
st.sidebar.markdown("**Expected numeric features**")
st.sidebar.caption("Age, AnnualIncome_k, SpendingScore, PurchasesPerMonth")
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded {len(raw)} rows")
else:
    raw = make_demo_data()
    st.sidebar.info("Using generated demo dataset (500 customers).")

st.sidebar.header("Clustering settings")
default_features = [c for c in ["Age", "AnnualIncome_k", "SpendingScore", "PurchasesPerMonth"] if c in raw.columns]
numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
features = st.sidebar.multiselect("Features", numeric_cols, default=default_features or numeric_cols[:4])
k = st.sidebar.slider("Number of clusters", min_value=2, max_value=8, value=4)
method = st.sidebar.radio("Algorithm", ["Compare both", "Agglomerative", "Divisive"])

if len(features) < 2:
    st.warning("Select at least two numeric features in the sidebar.")
    st.stop()

data = raw[features].replace([np.inf, -np.inf], np.nan)
valid = data.dropna().index
data = data.loc[valid]
if len(data) <= k:
    st.error("Please provide more valid rows than the selected number of clusters.")
    st.stop()

scaler = StandardScaler()
X = scaler.fit_transform(data)

results = {}
methods = ["Agglomerative", "Divisive"] if method == "Compare both" else [method]
for name in methods:
    start = time.perf_counter()
    if name == "Agglomerative":
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    else:
        labels = divisive_clustering(X, k)
    elapsed = time.perf_counter() - start
    score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
    results[name] = {"labels": labels, "time": elapsed, "silhouette": score}

tabs = st.tabs(["📊 Dashboard", "🌳 Dendrogram", "📋 Data & Results", "ℹ️ Methodology"])
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers analyzed", f"{len(data):,}")
    c2.metric("Selected clusters", str(k))
    best = max(results, key=lambda n: results[n]["silhouette"] if np.isfinite(results[n]["silhouette"]) else -1)
    c3.metric("Higher silhouette score", best)
    st.subheader("Algorithm comparison")
    comparison = pd.DataFrame([
        {"Algorithm": name, "Silhouette Score": result["silhouette"], "Execution Time (seconds)": result["time"]}
        for name, result in results.items()
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    if len(results) == 2:
        left, right = st.columns(2)
        for col, (name, result) in zip([left, right], results.items()):
            with col:
                st.markdown(f"#### {name}")
                plot_df = pd.DataFrame(X, columns=features)
                if len(features) == 2:
                    plot_df["Axis 1"], plot_df["Axis 2"] = X[:, 0], X[:, 1]
                else:
                    pca = PCA(n_components=2, random_state=42)
                    reduced = pca.fit_transform(X)
                    plot_df["Axis 1"], plot_df["Axis 2"] = reduced[:, 0], reduced[:, 1]
                plot_df["Cluster"] = result["labels"].astype(str)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=plot_df, x="Axis 1", y="Axis 2", hue="Cluster", palette="tab10", s=35, alpha=.8, ax=ax)
                ax.set_title(f"{name} clusters")
                if len(features) > 2:
                    ax.set_xlabel("PCA component 1")
                    ax.set_ylabel("PCA component 2")
                st.pyplot(fig)
                plt.close(fig)
    else:
        name = next(iter(results))
        result = results[name]
        plot_df = pd.DataFrame(X, columns=features)
        if len(features) == 2:
            plot_df["Axis 1"], plot_df["Axis 2"] = X[:, 0], X[:, 1]
        else:
            reduced = PCA(n_components=2, random_state=42).fit_transform(X)
            plot_df["Axis 1"], plot_df["Axis 2"] = reduced[:, 0], reduced[:, 1]
        plot_df["Cluster"] = result["labels"].astype(str)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=plot_df, x="Axis 1", y="Axis 2", hue="Cluster", palette="tab10", s=40, ax=ax)
        ax.set_title(f"{name} clusters")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Cluster profiles")
    for name, result in results.items():
        prof = data.copy()
        prof["Cluster"] = result["labels"]
        st.markdown(f"**{name}**")
        st.dataframe(prof.groupby("Cluster")[features].mean().round(2), use_container_width=True)

with tabs[1]:
    st.markdown("The dendrogram shows the **bottom-up merging process** for agglomerative clustering.")
    max_rows = min(len(X), 250)
    sample_idx = np.linspace(0, len(X)-1, max_rows, dtype=int)
    Z = linkage(X[sample_idx], method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=9, show_contracted=True, ax=ax)
    ax.set_title(f"Agglomerative Dendrogram (sample of {max_rows} customers)")
    ax.set_xlabel("Customer groups / leaves")
    ax.set_ylabel("Merge distance")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("For readability, the dendrogram uses at most 250 evenly sampled rows and truncates the display to the final 30 branches.")

with tabs[2]:
    st.subheader("Dataset preview")
    st.dataframe(raw.head(20), use_container_width=True)
    st.subheader("Cluster assignments")
    output = raw.loc[data.index].copy()
    for name, result in results.items():
        output[f"{name}_Cluster"] = result["labels"] + 1
    st.dataframe(output, use_container_width=True)
    st.download_button("Download cluster assignments CSV", output.to_csv(index=False).encode("utf-8"), "customer_cluster_results.csv", "text/csv")

with tabs[3]:
    st.markdown("""
    ### Agglomerative clustering (bottom-up)
    1. Start with every customer as an individual cluster.
    2. Find the two closest clusters.
    3. Merge them repeatedly until the requested number of clusters remains.

    ### Divisive clustering (top-down)
    1. Start with all customers in one cluster.
    2. Split the largest splittable cluster into two using K-means with `k=2`.
    3. Repeat until the requested number of clusters is reached.

    ### Preprocessing and evaluation
    - StandardScaler puts features on comparable scales.
    - PCA is used only to display higher-dimensional clusters in 2D.
    - Silhouette score ranges from -1 to 1; higher values generally indicate better-separated clusters.
    - Execution time depends on dataset size and hardware.

    **Limitation:** Divisive clustering here is a simple educational top-down heuristic, not an exact implementation of every divisive hierarchical clustering method. Silhouette score alone does not determine whether customer groups are commercially meaningful.
    """)
