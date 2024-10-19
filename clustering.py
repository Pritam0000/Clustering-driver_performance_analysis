import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_elbow_method(X, k_range):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots()
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    st.pyplot(fig)

def plot_kmeans_clusters(X, labels, feature_names):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('K-means Clustering Results')
    plt.colorbar(scatter)
    st.pyplot(fig)

def perform_clustering(df):
    st.header("Clustering Analysis")

    # Prepare data for clustering
    features = ['ctc', 'years_of_experience', 'salary_growth_rate', 'job_position_encoded']
    X = df[features].copy()

    # Remove any remaining infinite values
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    st.subheader("Elbow Method")
    k_range = range(1, 11)
    plot_elbow_method(X_scaled, k_range)

    # K-means Clustering
    st.subheader("K-means Clustering")
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Add cluster labels to the dataframe
    df.loc[X.index, 'Cluster'] = cluster_labels

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot clusters
    plot_kmeans_clusters(X_pca, cluster_labels, ['PCA1', 'PCA2'])

    # Display cluster statistics
    st.subheader("Cluster Statistics")
    for cluster in range(n_clusters):
        st.write(f"Cluster {cluster}:")
        st.write(df[df['Cluster'] == cluster][features].describe())

    # Analyze clusters
    st.subheader("Cluster Analysis")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df.loc[X.index, 'years_of_experience'], df.loc[X.index, 'ctc'], c=df.loc[X.index, 'Cluster'], cmap='viridis')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('CTC')
    ax.set_title('Clusters: Years of Experience vs CTC')
    plt.colorbar(scatter)
    st.pyplot(fig)

    # Job position distribution in clusters
    st.subheader("Job Position Distribution in Clusters")
    cluster_job_dist = df.loc[X.index].groupby('Cluster')['job_position'].value_counts(normalize=True).unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_job_dist.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Job Position Distribution in Clusters')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportion')
    plt.legend(title='Job Position', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)