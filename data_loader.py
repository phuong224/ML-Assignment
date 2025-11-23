# src/data_loader.py
import pandas as pd

def load_dataset(path="dataset/Dataset_5971.csv"):
    """
    Trả về X (pd.Series) và y (pd.Series lowercase)
    """
    df = pd.read_csv(path)
    if 'TEXT' not in df.columns or 'LABEL' not in df.columns:
        raise ValueError("Dataset must contain 'TEXT' and 'LABEL' columns.")
    X = df['TEXT']
    y = df['LABEL'].str.lower()
    return X, y
