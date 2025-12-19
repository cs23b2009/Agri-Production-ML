import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.data.loader import load_environmental_data, load_yield_data, get_merged_dataset
from src.models.registry import get_model_pipeline
from src.utils.constants import TARGET_COL, MODELS_DIR, RANDOM_STATE

def run_training_pipeline():
    # 1. Load and merge data
    print("Loading data...")
    env_df = load_environmental_data()
    yield_df = load_yield_data()
    df = get_merged_dataset(env_df, yield_df)
    
    # 2. Prepare features and target
    print("Preparing features...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=df['Crop']
    )
    
    # 3. Train models
    models_to_train = [
        "LinearRegression", "RandomForest", "XGBoost", 
        "DecisionTree", "LinearSVR", "GradientBoosting"
    ]
    
    results = []
    best_r2 = -np.inf
    best_model = None
    best_name = ""
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for name in models_to_train:
        print(f"Training {name}...")
        pipe = get_model_pipeline(name)
        pipe.fit(X_train, y_train)
        
        pred = pipe.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        results.append({"model": name, "r2": r2, "mae": mae, "rmse": rmse})
        print(f"{name} => R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipe
            best_name = name
            
    # 4. Save best model
    if best_model:
        model_path = os.path.join(MODELS_DIR, f"best_model_{best_name}.joblib")
        joblib.dump(best_model, model_path)
        print(f"Best model {best_name} saved to {model_path}")
        
    # 5. Save metrics
    metrics_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    run_training_pipeline()
