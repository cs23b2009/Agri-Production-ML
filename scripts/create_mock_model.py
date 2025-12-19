import joblib
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.utils.constants import MODELS_DIR, CAT_COLS, NUM_COLS

def create_mock_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model_LinearRegression.joblib")
    
    # Create a dummy pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ],
        remainder="drop",
    )
    
    pipe = Pipeline([
        ("pre", preprocessor),
        ("lin", LinearRegression())
    ])
    
    # Create dummy data to fit
    n_samples = 10
    X_dummy = pd.DataFrame({
        'N': np.random.rand(n_samples) * 100,
        'P': np.random.rand(n_samples) * 100,
        'K': np.random.rand(n_samples) * 100,
        'temperature': np.random.rand(n_samples) * 40,
        'humidity': np.random.rand(n_samples) * 100,
        'ph': np.random.rand(n_samples) * 14,
        'rainfall': np.random.rand(n_samples) * 1000,
        'Crop': ['coconut'] * n_samples,
        'Season': ['Whole Year'] * n_samples,
        'Soil_Type': ['loamy'] * n_samples
    })
    y_dummy = np.random.rand(n_samples) * 10
    
    pipe.fit(X_dummy, y_dummy)
    joblib.dump(pipe, model_path)
    print(f"Mock model created at {model_path}")

if __name__ == "__main__":
    create_mock_model()
