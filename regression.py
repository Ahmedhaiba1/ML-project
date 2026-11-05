import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD DATA & HANDLE MISSING VALUES / CLEANING
DATA_PATH = "data/raw/H2.csv"

# Load data: treat 'NULL', 'NA', empty strings as NaN
df = pd.read_csv(DATA_PATH, na_values=["NULL", "null", "NA", "na", ""], skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]

# Strip string columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "NULL": np.nan, "": np.nan})

# Prune extreme data-entry anomaly (ADR = $5,400) and negative values
df = df[(df["ADR"] >= 0) & (df["ADR"] <= 1000)].copy()


# 2. FEATURE ENGINEERING & LEAKAGE AUDIT
# Domain features known at booking time
df["TotalStayNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
df["TotalGuests"] = df["Adults"] + df["Children"].fillna(0) + df["Babies"]

# STRICT LEAKAGE PREVENTION:
# - Drop ReservationStatus & ReservationStatusDate (post-booking outcomes)
# - Drop AssignedRoomType (resolved at check-in, not quote time)
# - Drop IsCanceled (classification target, not available for ADR quote)
LEAKAGE_COLS = ["ReservationStatus", "ReservationStatusDate", "AssignedRoomType", "IsCanceled", "ADR"]

feature_cols = [c for c in df.columns if c not in LEAKAGE_COLS]
X = df[feature_cols]
y = df["ADR"]

# Train/Test Split (80% train, 20% test with fixed seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. MODULAR PREPROCESSING PIPELINE
num_cols = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "BookingChanges", "Agent", "Company", "DaysInWaitingList",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
    "TotalStayNights", "TotalGuests"
]

cat_cols = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment",
    "DistributionChannel", "ReservedRoomType", "DepositType", "CustomerType"
]

# Numerical: median imputation + standard scaling
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: constant imputation + one-hot encoding (infrequent levels grouped)
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# Baseline Model: Linear Regression
baseline_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# 4. 5-FOLD CROSS-VALIDATION (ON TRAINING SET)
print("Running 5-Fold Cross-Validation on training data...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}
cv_results = cross_validate(baseline_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

print(f"CV 5-Fold MAE:  ${-cv_results['test_mae'].mean():.2f} (+/- {cv_results['test_mae'].std():.2f})")
print(f"CV 5-Fold RMSE: ${-cv_results['test_rmse'].mean():.2f} (+/- {cv_results['test_rmse'].std():.2f})")
print(f"CV 5-Fold R²:   {cv_results['test_r2'].mean():.4f} (+/- {cv_results['test_r2'].std():.4f})")

# 5. TRAIN ON FULL TRAIN SET & EVALUATE ON TEST SET

print("\nFitting final model on full training set...")
baseline_pipeline.fit(X_train, y_train)

y_pred = baseline_pipeline.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print("\n=== FINAL TEST SET RESULTS (Baseline Linear Regression) ===")
print(f"Test MAE:  ${test_mae:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")
print(f"Test R²:   {test_r2:.4f}")


# 6. SAVE ARTIFACT FOR INFERENCE

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(baseline_pipeline, "artifacts/regression_linear_pipeline.joblib")
print("\nSaved pipeline to artifacts/regression_linear_pipeline.joblib")
