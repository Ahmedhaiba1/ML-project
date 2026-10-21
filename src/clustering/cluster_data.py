"""
Clustering - data preparation (Step 2: cleaning, Step 3: encoding + scaling).

Place this file in:  src/clustering/cluster_data.py

Analysis point: the moment the booking is made. Only features known at that
time are used, and no outcome column (IsCanceled, ReservationStatus, ...) is kept.
The original H2.csv is never modified.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------- paths
def find_project_dir() -> Path:
    """Walk up from this file until a folder containing data/H2.csv is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "H2.csv").exists():
            return parent
    raise FileNotFoundError("Could not find data/H2.csv above this file.")


PROJECT_DIR = find_project_dir()
DATA_FILE = PROJECT_DIR / "data" / "H2.csv"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "clustering"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "clustering"

# ---------------------------------------------------------------- features
LOG_NUMERIC = ["LeadTime", "TotalNights", "ADR"]          # skewed -> log1p + scale
PLAIN_NUMERIC = ["Adults", "Children", "Babies"]           # scale only
NUMERIC_FEATURES = LOG_NUMERIC + PLAIN_NUMERIC
CATEGORICAL_FEATURES = [
    "MarketSegment",
    "DistributionChannel",
    "CustomerType",
    "DepositType",
    "Meal",
    "ReservedRoomType",
    "ArrivalDateMonth",
]
CLUSTER_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Cleaning thresholds
ADR_MAX = 500
BABIES_MAX = 4  # rows with 5 or more babies are treated as data errors


# ---------------------------------------------------------------- loading
def load_raw() -> pd.DataFrame:
    """Read the original dataset (no changes)."""
    return pd.read_csv(DATA_FILE)


# ---------------------------------------------------------------- cleaning
def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return a cleaned COPY of the raw data. Each step reports rows removed."""
    df = df.copy()
    log = [("Original rows", len(df))]

    # 1. Exact duplicates
    df = df.drop_duplicates()
    log.append(("After removing exact duplicates", len(df)))

    # 2. Children is stored as text (has missing values) -> numeric, missing = 0
    df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0).astype(int)

    # 3. Strip whitespace in text columns and treat "NULL" as missing
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"NULL": pd.NA, "": pd.NA})

    # 4. Derived feature
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]

    # 5. Invalid stays: zero nights
    df = df[df["TotalNights"] > 0]
    log.append(("After removing zero-night stays", len(df)))

    # 6. Invalid bookings: no guests at all
    df = df[(df["Adults"] + df["Children"] + df["Babies"]) > 0]
    log.append(("After removing zero-guest bookings", len(df)))

    # 7. Impossible / extreme ADR (negative, or above threshold)
    df = df[(df["ADR"] >= 0) & (df["ADR"] <= ADR_MAX)]
    log.append((f"After removing ADR < 0 or > {ADR_MAX}", len(df)))

    # 8. Data-error babies count
    df = df[df["Babies"] <= BABIES_MAX]
    log.append((f"After removing Babies > {BABIES_MAX}", len(df)))

    # Remaining categorical missing values (rare) -> explicit "Unknown"
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    df = df.reset_index(drop=True)

    if verbose:
        print("\nCLEANING SUMMARY")
        print("-" * 50)
        for name, n in log:
            print(f"{name:<40}{n:>10,}")
    return df


def get_cluster_frame(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Keep only the clustering features (no outcome columns)."""
    return df_clean[CLUSTER_FEATURES].copy()


# ---------------------------------------------------------------- preprocessing
def build_preprocessor() -> ColumnTransformer:
    """log1p + scale for skewed numerics, scale for others, one-hot for categoricals."""
    log_scale = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("log_num", log_scale, LOG_NUMERIC),
            ("num", StandardScaler(), PLAIN_NUMERIC),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,  # categories under 1% -> "infrequent" (Other)
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def prepare_features(save: bool = True):
    """
    Full Step 2 + 3: load -> clean -> select features -> fit preprocessor -> transform.
    Returns (df_clean, X_matrix, feature_names, fitted_preprocessor).
    """
    df_clean = clean_data(load_raw())
    frame = get_cluster_frame(df_clean)

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(frame)
    feature_names = preprocessor.get_feature_names_out().tolist()

    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, ARTIFACT_DIR / "cluster_preprocessor.joblib")
        df_clean.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)

    return df_clean, X, feature_names, preprocessor


def load_preprocessor() -> ColumnTransformer:
    """Load the saved preprocessor (used later for transforming new bookings)."""
    return joblib.load(ARTIFACT_DIR / "cluster_preprocessor.joblib")


if __name__ == "__main__":
    df_clean, X, names, _ = prepare_features()
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Numeric features: {NUMERIC_FEATURES}")
    print(f"Encoded columns: {len(names)}")
    print(f"Saved preprocessor to: {ARTIFACT_DIR}")