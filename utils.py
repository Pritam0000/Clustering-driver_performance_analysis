import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_elbow_method(X, k_range):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(k_range, inertias, 'bx-')
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    st.pyplot(fig)

def plot_kmeans_clusters(X, labels, feature_names):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('K-means Clustering Results')
    plt.colorbar(scatter)
    st.pyplot(fig)

    