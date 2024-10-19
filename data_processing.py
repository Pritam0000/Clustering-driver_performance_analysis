import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    # Load the data
    df = pd.read_csv('scaler_clustering.csv')

    # Handle missing values
    df['orgyear'].fillna(df['orgyear'].median(), inplace=True)
    df['ctc'].fillna(df['ctc'].median(), inplace=True)
    df['ctc_updated_year'].fillna(df['ctc_updated_year'].median(), inplace=True)
    df['job_position'].fillna('Unknown', inplace=True)

    # Convert year columns to int
    year_columns = ['orgyear', 'ctc_updated_year']
    for col in year_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median()).astype(int)

    # Calculate years of experience
    current_year = 2024
    df['years_of_experience'] = current_year - df['orgyear']
    df['years_of_experience'] = df['years_of_experience'].clip(lower=0)  # Ensure non-negative values

    # Calculate salary growth rate
    df['salary_growth_rate'] = (df['ctc'] / df['years_of_experience']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Handle extreme values
    for col in ['ctc', 'salary_growth_rate']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Encode categorical variables
    le = LabelEncoder()
    df['job_position_encoded'] = le.fit_transform(df['job_position'])

    # Create a binary column for high earners
    df['is_high_earner'] = (df['ctc'] > df['ctc'].median()).astype(int)

    return df