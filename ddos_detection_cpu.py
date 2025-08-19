import pandas as pd
import numpy as np
import platform
import psutil
import time
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def print_system_info():
    """Print CPU and system information"""
    cpu_info = platform.processor()
    cpu_count = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()

    print(f"CPU: {cpu_info}")
    print(f"Physical Cores: {cpu_count}")
    print(f"Logical Cores: {logical_cores}")
    print(f"Max Frequency: {cpu_freq.max:.2f} MHz")
    print(f"Min Frequency: {cpu_freq.min:.2f} MHz")
    print(f"Current Frequency: {cpu_freq.current:.2f} MHz")
    print(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")

def load_and_prepare_data(file_path):
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv(file_path)
        
        print("Dataset Head:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        
        target_column = df.columns[-1]  # Assuming last column is target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        numeric_cols = X.select_dtypes(include=np.number).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].astype(str)
        
        return X, y, categorical_cols
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def train_and_evaluate_model(X, y, categorical_cols):
    """Train and evaluate the CatBoost model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_pool = Pool(X_train, label=y_train, cat_features=categorical_cols)
    test_pool = Pool(X_test, label=y_test, cat_features=categorical_cols)
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        task_type='CPU'  # Use CPU
    )
    
    start_time = time.time()
    model.fit(train_pool, eval_set=test_pool)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print("\nModel Performance:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def main():
    print_system_info()
    
    file_path = "/Users/divjyotkaur/Desktop/project/DDOSCB.csv"  # Updated path for macOS
    X, y, categorical_cols = load_and_prepare_data(file_path)
    
    if X is None or y is None:
        return
    
    model = train_and_evaluate_model(X, y, categorical_cols)
    
    model.save_model('ddos_detection_model_cpu.cbm')
    print("\nModel saved successfully as 'ddos_detection_model_cpu.cbm'")

if __name__ == "__main__":
    main()
