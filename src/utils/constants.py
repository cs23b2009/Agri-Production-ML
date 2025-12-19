import os

# Column Names
TARGET_COL = "Yield_t_ha"
CAT_COLS = ["Crop", "Season", "Soil_Type"]
NUM_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Model Names
LINEAR_REGRESSION = "LinearRegression"
RANDOM_FOREST = "RandomForest"
XGBOOST = "XGBoost"

# Config
RANDOM_STATE = 42

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models_prod")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
