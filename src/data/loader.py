import pandas as pd
import os
from src.utils.constants import RAW_DATA_DIR
from src.utils.mappings import CROP_MAP, CROP_SOIL_MAP

def load_environmental_data(filename="Crop_recommendation.csv"):
    path = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Environmental data not found at {path}")
    df = pd.read_csv(path)
    df['label'] = df['label'].str.strip().str.lower()
    return df

def load_yield_data(filename="crop_yield.csv"):
    path = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Yield data not found at {path}")
    df = pd.read_csv(path)
    return df

def clean_yield_data(yield_df):
    # Select columns
    df = yield_df[["Crop", "Season", "Yield"]].copy()
    
    # Remove outliers using IQR
    Q1 = df['Yield'].quantile(0.25)
    Q3 = df['Yield'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Yield'] >= lower_bound) & (df['Yield'] <= upper_bound)]
    
    # Standardize crop names
    df['Crop'] = df['Crop'].str.strip().str.lower()
    df['Crop'] = df['Crop'].replace(CROP_MAP)
    
    # Map soil types
    df['Soil_Type'] = df['Crop'].map(CROP_SOIL_MAP)
    
    return df

def get_merged_dataset(env_df, yield_df):
    yield_clean = clean_yield_data(yield_df)
    merged_df = pd.merge(yield_clean, env_df, left_on='Crop', right_on='label', how='left')
    # Drop redundant column
    if 'label' in merged_df.columns:
        merged_df = merged_df.drop(columns=['label'])
    return merged_df
