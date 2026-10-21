
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Locate project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "H2.csv"

# 2. Read original dataset (no cleaning)
df = pd.read_csv(DATA_FILE)

# 3. Create folder for EDA results
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("EDA - EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 4. Dataset overview
print("\nDATASET SHAPE")
print(df.shape)

print("\nDATA TYPES")
print(df.dtypes)

# 5. Missing values
print("\nMISSING VALUES")
print(df.isna().sum().sort_values(ascending=False))

# 6. Numerical summary
numerical_columns = df.select_dtypes(include="number").columns

print("\nNUMERICAL SUMMARY")
print(df[numerical_columns].describe().T.round(2))

# 7. Categorical summary
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("\nCATEGORICAL CARDINALITY")
print(df[categorical_columns].nunique().sort_values(ascending=False))

# 8. Plot numerical distributions
features = [
    "LeadTime",
    "ADR",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "TotalOfSpecialRequests",
]

for feature in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=feature, bins=40)
    plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{feature}_histogram.png")
    plt.close()

# 9. Plot boxplots to investigate outliers
for feature in ["ADR", "LeadTime"]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=feature)
    plt.title(f"Boxplot of {feature}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{feature}_boxplot.png")
    plt.close()

# 10. Categorical frequency tables
for feature in ["Meal", "MarketSegment", "CustomerType", "DepositType"]:
    print(f"\nTOP VALUES IN {feature}")
    print(df[feature].value_counts(dropna=False).head(15))

# 11. Save summary tables
df[numerical_columns].describe().T.to_csv(
    OUTPUT_DIR / "numerical_summary.csv"
)

df[categorical_columns].nunique().to_csv(
    OUTPUT_DIR / "categorical_cardinality.csv",
    header=["unique_values"]
)

print("\nEDA completed successfully.")
print(f"Results saved in: {OUTPUT_DIR}")
