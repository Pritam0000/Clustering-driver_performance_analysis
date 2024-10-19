import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    st.header("Exploratory Data Analysis")

    # Display basic information about the dataset
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Shape of the dataset: {df.shape}")

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Univariate Analysis
    st.subheader("Univariate Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='years_of_experience', y='ctc', hue='job_position')
    ax.set_title('Years of Experience vs CTC')
    plt.legend(title='Job Position', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Job Position Analysis
    st.subheader("Job Position Analysis")
    job_position_counts = df['job_position'].value_counts()
    fig, ax = plt.subplots()
    job_position_counts.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Job Positions')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Salary Growth Rate Analysis
    st.subheader("Salary Growth Rate Analysis")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='job_position', y='salary_growth_rate', ax=ax)
    ax.set_title('Salary Growth Rate by Job Position')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)