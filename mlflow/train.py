import mlflow
import mlflow.xgboost
import mlflow.catboost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

def train_mlflow(df, model_type="xgboost", classification_type="multiclass"):
    
    # 1. Define Split Dates (From Notebook)
    last_train_date = pd.to_datetime('2015-07-31 01:00:00')
    first_test_date = pd.to_datetime('2015-08-01 01:00:00')

    # 2. Prepare X and y
    y_train = df.loc[df['datetime'] < last_train_date, 'failure']
    X_train = df.loc[df['datetime'] < last_train_date].drop(['datetime', 'failure'], axis=1)
    
    y_test = df.loc[df['datetime'] >= first_test_date, 'failure']
    X_test = df.loc[df['datetime'] >= first_test_date].drop(['datetime', 'failure'], axis=1)

    # 3. Handle Labels (Multi vs Binary)
    if classification_type == "binary":
        # Convert all failures to 1, none to 0
        dict_map = {'none': 0, 'comp1': 1, 'comp2': 1, 'comp3': 1, 'comp4': 1}
    else:
        # Multiclass Mapping
        dict_map = {'none': 0, 'comp1': 1, 'comp2': 2, 'comp3': 3, 'comp4': 4}

    y_train = y_train.map(dict_map).astype(int)
    y_test = y_test.map(dict_map).astype(int)

    # 4. Start MLflow Run
    experiment_name = f"PdM_{model_type}_{classification_type}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        print(f" Starting Training: {experiment_name}")
        
        #  MODEL SELECTION 
        if model_type == "xgboost":
            objective = 'binary:logistic' if classification_type == "binary" else 'multi:softprob'
            model = XGBClassifier(objective=objective, n_estimators=50) # Reduced estimators for speed
            mlflow.log_param("model_type", "XGBoost")
        
        elif model_type == "catboost":
            loss_function = 'Logloss' if classification_type == "binary" else 'MultiClass'
            model = CatBoostClassifier(
                iterations=500, 
                learning_rate=0.1, 
                depth=6, 
                loss_function=loss_function,
                auto_class_weights='Balanced',
                verbose=0
            )
            mlflow.log_param("model_type", "CatBoost")

        #  TRAINING 
        model.fit(X_train, y_train)
        
        #  PREDICTION 
        preds = model.predict(X_test)
        
        # Catboost sometimes returns [[0], [1]] arrays, flatten them
        if model_type == "catboost":
            preds = [x[0] for x in preds]

        #  METRICS 
        average_method = 'binary' if classification_type == "binary" else 'macro'
        
        precision = precision_score(y_test, preds, average=average_method)
        recall = recall_score(y_test, preds, average=average_method)
        f1 = f1_score(y_test, preds, average=average_method)

        print(f" Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        #  LOGGING TO MLFLOW 
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log feature names list
        feature_names = X_train.columns.tolist()
        mlflow.log_dict({"features": feature_names}, "features.json")

        # Log Model
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.catboost.log_model(model, "model")

        print(f" Run Complete. Saved to MLflow.")
