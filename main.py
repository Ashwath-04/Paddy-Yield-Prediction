import os
import joblib
from sklearn.model_selection import train_test_split

from src.config import TARGET, DROP_COLUMNS
from src.data_loader import load_data
from src.data_preprocessing import handle_missing, encode_categorical
from src.feature_engineering import create_features
from src.feature_selection import select_features
from src.train_lgbm import train_lgbm
from src.train_xgb import train_xgb
from src.evaluate import evaluate
from src.tune_lgbm_optuna import tune_lgbm
from src.tune_xgb_optuna import tune_xgb
from src.explainability import explain_model
from src.optuna_visual import save_optuna_plots
from src.eda import basic_info, plot_target_distribution,correlation_heatmap
from src.normalise import remove_outliers_zscore

# Load dataset
df = load_data("data/raw/paddydataset.csv")
basic_info(df)
plot_target_distribution(df, "Paddy_yield(in_Kg)")
correlation_heatmap(df)
print("Columns:", df.columns)

# Preprocessing
df = handle_missing(df)
df = encode_categorical(df)

# Remove outliers using Z-score
df = remove_outliers_zscore(df)

# Feature Engineering
df = create_features(df)

# Drop useless columns (low information)
drop_cols = [
    'Nursery',
    'Soil_Types',
    'Wind_Direction_D1_D30',
    'Wind_Direction_D31_D60',
    'Wind_Direction_D61_D90',
    'Wind_Direction_D91_D120'
]

df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

# Define target
target = TARGET

X = df.drop(target, axis=1)
y = df[target]


os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/paddy_processed.csv", index=False)

# Feature Selection
X = select_features(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# AFTER train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply normalization
#X_train, X_test = normalize_data(X_train, X_test)
os.makedirs("outputs/plots", exist_ok=True)

# Train tuned models
print("\nTuning LightGBM...")
lgb_model, lgb_study = tune_lgbm(X_train, y_train)

print("\nTuning XGBoost...")
xgb_model, xgb_study = tune_xgb(X_train, y_train)

# Evaluate
print("\nLightGBM Results:")
evaluate(lgb_model, X_test, y_test)

print("\nXGBoost Results:")
evaluate(xgb_model, X_test, y_test)

# Save Optuna plots
save_optuna_plots(lgb_study, "lgbm")
save_optuna_plots(xgb_study, "xgb")

# SHAP explainability
print("\nGenerating SHAP plots...")
explain_model(lgb_model, X_test, None)

os.makedirs("outputs/models", exist_ok=True)

# Save models
joblib.dump(lgb_model, "outputs/models/lgbm_model.pkl")
joblib.dump(xgb_model, "outputs/models/xgb_model.pkl")

os.makedirs("outputs/reports", exist_ok=True)

evaluate(lgb_model, X_test, y_test, "lgbm")
evaluate(xgb_model, X_test, y_test, "xgb")