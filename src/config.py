"""Central configuration: paths, column schema, seeds and search budgets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RAW_FILE = DATA_RAW / "paddydataset.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/1186/paddy+dataset.zip"
DATASET_DOI = "10.24432/C55W3J"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
OPTUNA_TRIALS = 40

# Raw column name -> clean snake_case name. Units follow the source headers.
COLUMN_MAP = {
    "Hectares": "hectares",
    "Agriblock": "agriblock",
    "Variety": "variety",
    "Soil Types": "soil_type",
    "Seedrate(in Kg)": "seed_rate_kg",
    "LP_Mainfield(in Tonnes)": "lp_mainfield_t",
    "Nursery": "nursery_type",
    "Nursery area (Cents)": "nursery_area_cents",
    "LP_nurseryarea(in Tonnes)": "lp_nursery_t",
    "DAP_20days": "dap_20d",
    "Weed28D_thiobencarb": "thiobencarb_28d",
    "Urea_40Days": "urea_40d",
    "Potassh_50Days": "potash_50d",
    "Micronutrients_70Days": "micronutrients_70d",
    "Pest_60Day(in ml)": "pesticide_60d_ml",
    "30DRain( in mm)": "rain_d0_30_mm",
    "30DAI(in mm)": "ai_d0_30_mm",
    "30_50DRain( in mm)": "rain_d30_50_mm",
    "30_50DAI(in mm)": "ai_d30_50_mm",
    "51_70DRain(in mm)": "rain_d51_70_mm",
    "51_70AI(in mm)": "ai_d51_70_mm",
    "71_105DRain(in mm)": "rain_d71_105_mm",
    "71_105DAI(in mm)": "ai_d71_105_mm",
    "Min temp_D1_D30": "tmin_d1_30",
    "Max temp_D1_D30": "tmax_d1_30",
    "Min temp_D31_D60": "tmin_d31_60",
    "Max temp_D31_D60": "tmax_d31_60",
    "Min temp_D61_D90": "tmin_d61_90",
    "Max temp_D61_D90": "tmax_d61_90",
    "Min temp_D91_D120": "tmin_d91_120",
    "Max temp_D91_D120": "tmax_d91_120",
    "Inst Wind Speed_D1_D30(in Knots)": "wind_kn_d1_30",
    "Inst Wind Speed_D31_D60(in Knots)": "wind_kn_d31_60",
    "Inst Wind Speed_D61_D90(in Knots)": "wind_kn_d61_90",
    "Inst Wind Speed_D91_D120(in Knots)": "wind_kn_d91_120",
    "Wind Direction_D1_D30": "wind_dir_d1_30",
    "Wind Direction_D31_D60": "wind_dir_d31_60",
    "Wind Direction_D61_D90": "wind_dir_d61_90",
    "Wind Direction_D91_D120": "wind_dir_d91_120",
    "Relative Humidity_D1_D30": "rh_d1_30",
    "Relative Humidity_D31_D60": "rh_d31_60",
    "Relative Humidity_D61_D90": "rh_d61_90",
    "Relative Humidity_D91_D120": "rh_d91_120",
    "Trash(in bundles)": "trash_bundles",
    "Paddy yield(in Kg)": "yield_kg",
}

TARGET_RAW = "yield_kg"            # total paddy yield per farm record (kg)
TARGET_PER_HA = "yield_kg_per_ha"  # derived: yield_kg / hectares (kg/ha)
TARGETS = {"total": TARGET_RAW, "per_ha": TARGET_PER_HA}

# Post-harvest by-product: only observable after harvest -> excluded from features.
LEAKAGE_COLUMNS = ["trash_bundles"]

STAGES = ["d1_30", "d31_60", "d61_90", "d91_120"]
RAIN_STAGES = ["d0_30", "d30_50", "d51_70", "d71_105"]

INPUT_QTY_COLUMNS = [
    "seed_rate_kg", "lp_mainfield_t", "nursery_area_cents", "lp_nursery_t",
    "dap_20d", "thiobencarb_28d", "urea_40d", "potash_50d",
    "micronutrients_70d", "pesticide_60d_ml",
]
RAIN_COLUMNS = [f"rain_{s}_mm" for s in RAIN_STAGES]
AI_COLUMNS = [f"ai_{s}_mm" for s in RAIN_STAGES]
TMIN_COLUMNS = [f"tmin_{s}" for s in STAGES]
TMAX_COLUMNS = [f"tmax_{s}" for s in STAGES]
WIND_SPEED_COLUMNS = [f"wind_kn_{s}" for s in STAGES]
WIND_DIR_COLUMNS = [f"wind_dir_{s}" for s in STAGES]
RH_COLUMNS = [f"rh_{s}" for s in STAGES]
WEATHER_COLUMNS = (RAIN_COLUMNS + AI_COLUMNS + TMIN_COLUMNS + TMAX_COLUMNS
                   + WIND_SPEED_COLUMNS + WIND_DIR_COLUMNS + RH_COLUMNS)
CATEGORICAL_COLUMNS = ["agriblock", "variety", "soil_type", "nursery_type"]
GEO_COLUMNS = ["agriblock"]

# 16-point compass -> degrees, for cyclic encoding of wind direction.
COMPASS_DEG = {d: i * 22.5 for i, d in enumerate(
    ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])}

for _d in (DATA_PROCESSED, MODELS_DIR, FIG_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
