import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_processing import load_and_preprocess_data
from exploratory_analysis import perform_eda
from clustering import perform_clustering
from performance_metrics import analyze_performance

def main():
    st.title('Driver Performance Analysis')

    df = load_and_preprocess_data()

    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Data Overview', 'Exploratory Analysis', 'Performance Metrics', 'Clustering'])

    if page == 'Data Overview':
        st.header('Data Overview')
        st.write(df.head())
        st.write(df.describe())

    elif page == 'Exploratory Analysis':
        perform_eda(df)

    elif page == 'Performance Metrics':
        analyze_performance(df)

    elif page == 'Clustering':
        perform_clustering(df)

if __name__ == '__main__':
    main()