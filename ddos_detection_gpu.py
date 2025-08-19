import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

def check_gpu():
    """Check GPU availability and print information"""
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
    else:
        print("No GPU available. Running on CPU.")

def load_data(file_path):
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(file_path)
        print("Dataset Head:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_features(df):
    """Prepare features for the model"""
    categorical_columns = ['src', 'dst', 'Protocol']
    X = df.drop('label', axis=1)
    y = df['label']

    for col in categorical_columns:
        X[col] = X[col].astype(str)
    
    return X, y, categorical_columns

def train_model(X, y, cat_features):
    """Train the CatBoost model using GPU"""
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        task_type='GPU' if torch.cuda.is_available() else 'CPU'
    )
    model.fit(X, y, cat_features=cat_features)
    return model

def main():
    check_gpu()
    
    # Set correct file path for macOS
    file_path = "/Users/divjyotkaur/Desktop/project/DDOSCB.csv"
    df = load_data(file_path)
    
    if df is None:
        return
    
    X, y, cat_features = prepare_features(df)
    model = train_model(X, y, cat_features)
    
    model.save_model('ddos_detection_model_gpu.cbm')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
