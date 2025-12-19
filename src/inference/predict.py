import joblib
import pandas as pd
import os
import glob
from src.utils.constants import MODELS_DIR

def load_best_model():
    model_files = glob.glob(os.path.join(MODELS_DIR, "best_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError("No trained models found in models_prod/")
    # Load the most recent or first one
    return joblib.load(model_files[0])

def predict_yield(input_data):
    """
    input_data: dict containing keys:
    Crop, Season, Soil_Type, N, P, K, temperature, humidity, ph, rainfall
    """
    model = load_best_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]
