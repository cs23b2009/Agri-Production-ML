from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from src.utils.constants import CAT_COLS, NUM_COLS, RANDOM_STATE

def get_tree_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ],
        remainder="drop",
    )

def get_linear_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("scale", StandardScaler(), NUM_COLS),
        ],
        remainder="drop",
    )

def get_model_pipeline(model_name):
    if model_name == "LinearRegression":
        return Pipeline([
            ("pre", get_linear_preprocessor()),
            ("lin", LinearRegression())
        ])
    elif model_name == "RandomForest":
        return Pipeline([
            ("pre", get_tree_preprocessor()),
            ("rf", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42))
        ])
    elif model_name == "XGBoost":
        return Pipeline([
            ("pre", get_tree_preprocessor()),
            ("xgb", XGBRegressor(
                n_estimators=1000, learning_rate=0.05, max_depth=8,
                objective="reg:squarederror", random_state=42, n_jobs=-1
            ))
        ])
    elif model_name == "DecisionTree":
        return Pipeline([
            ("pre", get_tree_preprocessor()),
            ("dt", DecisionTreeRegressor(random_state=42))
        ])
    elif model_name == "LinearSVR":
        return Pipeline([
            ("pre", get_linear_preprocessor()),
            ("svr", LinearSVR(random_state=42, max_iter=10000))
        ])
    elif model_name == "GradientBoosting":
        return Pipeline([
            ("pre", get_tree_preprocessor()),
            ("gbr", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
        ])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
