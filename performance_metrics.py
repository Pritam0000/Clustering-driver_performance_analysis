import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_performance(df):
    st.header("Performance Metrics Analysis")

    # CTC Distribution
    st.subheader("CTC Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['ctc'], kde=True, ax=ax)
    ax.set_title('Distribution of CTC')
    ax.set_xlabel('CTC')
    st.pyplot(fig)

    # Average CTC by Job Position
    st.subheader("Average CTC by Job Position")
    avg_ctc = df.groupby('job_position')['ctc'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    avg_ctc.plot(kind='bar', ax=ax)
    ax.set_title('Average CTC by Job Position')
    ax.set_ylabel('Average CTC')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Salary Growth Rate Analysis
    st.subheader("Salary Growth Rate Analysis")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='years_of_experience', y='salary_growth_rate', hue='job_position', ax=ax)
    ax.set_title('Salary Growth Rate vs Years of Experience')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary Growth Rate')
    plt.legend(title='Job Position', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)