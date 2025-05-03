import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Paths
RAW_DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    return df

def preprocess_data(df):
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    # Column groups
    numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical = [col for col in X.columns if X[col].dtype == 'object']

    # Pipelines
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ])

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor

def split_and_save(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    joblib.dump(X_train, os.path.join(PROCESSED_DATA_DIR, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(PROCESSED_DATA_DIR, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(PROCESSED_DATA_DIR, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(PROCESSED_DATA_DIR, "y_test.pkl"))

def save_pipeline(preprocessor):
    joblib.dump(preprocessor, os.path.join(PROCESSED_DATA_DIR, "preprocessor.pkl"))

def main():
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    split_and_save(X, y)
    save_pipeline(preprocessor)
    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    main()