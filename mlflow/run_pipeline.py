from src.data_processing import load_and_process_data
from src.train import train_mlflow
import pandas as pd
import os

if __name__ == "__main__":
    #  Feature Engineering
    # 
    if os.path.exists("data/labeled_features_processed.csv"):
        print(" Found processed data. Loading...")
        df = pd.read_csv("data/labeled_features_processed.csv")
        # Ensure datetime is datetime object
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        # Run the heavy processing
        df = load_and_process_data(data_dir="data")
        # Save it for next time
        df.to_csv("data/labeled_features_processed.csv", index=False)

    # Run Experiments
    
    #  XGBoost Multiclass
    train_mlflow(df, model_type="xgboost", classification_type="multiclass")

    #  CatBoost Multiclass
    train_mlflow(df, model_type="catboost", classification_type="multiclass")

    #  CatBoost Binary
    train_mlflow(df, model_type="catboost", classification_type="binary")