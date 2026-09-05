import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define Rating Categories mapping
RATING_MAP = {
    'TV-MA': 'Adults', 'R': 'Adults', 'NC-17': 'Adults',
    'TV-14': 'Teens', 'PG-13': 'Teens',
    'TV-PG': 'Kids/Family', 'PG': 'Kids/Family', 'TV-Y': 'Kids/Family', 
    'TV-Y7': 'Kids/Family', 'G': 'Kids/Family', 'TV-Y7-FV': 'Kids/Family'
}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Netflix dataset:
    - Drops rows without rating or invalid rating entries
    - Groups target rating into 3 main categories: Adults, Teens, Kids/Family
    - Extracts duration numerical values
    """
    df = df.copy()
    
    # Filter valid ratings
    df = df[df['rating'].isin(RATING_MAP.keys())].copy()
    df['target_category'] = df['rating'].map(RATING_MAP)
    
    # Text missing values
    text_cols = ['title', 'director', 'country', 'listed_in']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = 'Unknown'
            
    # Duration processing
    if 'duration' in df.columns:
        df['duration_num'] = df['duration'].astype(str).str.extract(r'(\d+)').astype(float)
        df['duration_num'] = df['duration_num'].fillna(df['duration_num'].median())
    else:
        df['duration_num'] = 90.0
        
    return df

def build_preprocessor_pipeline():
    """
    Builds Scikit-Learn ColumnTransformer for text, categorical, and numerical features.
    """
    text_transformer = TfidfVectorizer(max_features=1000, stop_words='english')
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_genre', text_transformer, 'listed_in'),
            ('cat_type', cat_transformer, ['type']),
            ('num', num_transformer, ['release_year', 'duration_num'])
        ]
    )
    return preprocessor

def get_train_test_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets using stratification on target_category.
    """
    X = df[['type', 'release_year', 'duration_num', 'listed_in', 'country']]
    y = df['target_category']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
