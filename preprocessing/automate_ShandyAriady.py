import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape}")
    return df

def remove_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Duplicates removed: {before - after} rows")
    return df

def encode_features(df):
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    print(f"Encoded columns: {list(categorical_cols)}")
    return df

def scale_features(df, target_col='NObeyesdad'):
    scaler = MinMaxScaler()
    feature_cols = df.drop(target_col, axis=1).columns
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("Scaling done.")
    return df

def split_data(df, target_col='NObeyesdad', test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def main():
    # Path
    input_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
    output_path = 'preprocessing/ObesityDataSet_preprocessing.csv'

    # Pipeline
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = encode_features(df)
    df = scale_features(df)
    save_data(df, output_path)

    # Split (opsional disimpan terpisah)
    X_train, X_test, y_train, y_test = split_data(df)
    print("Preprocessing selesai!")

if __name__ == '__main__':
    main()