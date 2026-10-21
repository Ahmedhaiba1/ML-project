
import pandas as pd
from pathlib import Path

# ==================================================
# 1. LOAD DATASET
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "H2.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

# ==================================================
# 2. DATASET OVERVIEW
# ==================================================

print("\n1. DATASET OVERVIEW")
print("-" * 40)

print("Dataset shape:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

# ==================================================
# 3. MISSING VALUES
# ==================================================

print("\n2. MISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No actual missing values found.")
else:
    print(missing.sort_values(ascending=False))

# ==================================================
# 4. DUPLICATE ANALYSIS
# ==================================================

print("\n3. DUPLICATE ANALYSIS")
print("-" * 40)

print("Exact duplicate rows:", df.duplicated().sum())

print(
    "Rows belonging to duplicate groups:",
    df.duplicated(keep=False).sum()
)

print("Rows after removing exact duplicates (count only):",
      len(df.drop_duplicates()))

# ==================================================
# 5. TEXT PLACEHOLDER ANALYSIS
# ==================================================

print("\n4. TEXT PLACEHOLDERS")
print("-" * 40)

found_placeholder = False

for col in df.select_dtypes(include=["object", "string"]).columns:
    values = df[col].astype("string").str.strip().str.upper()

    count = values.isin(["NULL", "NA", "N/A", ""]).sum()

    if count > 0:
        print(f"{col}: {count}")
        found_placeholder = True

if not found_placeholder:
    print("No common text placeholders found.")

# Show the most common values in selected columns
print("\nCommon categorical values:")

for col in ["Agent", "Company", "Country", "Meal"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(10))

# ==================================================
# 6. ADR ANALYSIS
# ==================================================

print("\n5. ADR ANALYSIS")
print("-" * 40)

if "ADR" in df.columns:
    print("ADR values equal to zero:", (df["ADR"] == 0).sum())
    print("ADR values below zero:", (df["ADR"] < 0).sum())
    print("ADR values above 500:", (df["ADR"] > 500).sum())
    print("ADR values above 1000:", (df["ADR"] > 1000).sum())

    print("\nADR statistical summary:")
    print(df["ADR"].describe())

    print("\nHighest 10 ADR values:")
    print(df["ADR"].nlargest(10).to_list())

# ==================================================
# 7. CANCELLATION DISTRIBUTION
# ==================================================

print("\n6. CANCELLATION DISTRIBUTION")
print("-" * 40)

if "IsCanceled" in df.columns:
    print("\nCounts:")
    print(df["IsCanceled"].value_counts(dropna=False))

    print("\nPercentages:")
    print(
        (df["IsCanceled"].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )

# ==================================================
# 8. STAY LENGTH CHECK
# ==================================================

print("\n7. STAY LENGTH CHECK")
print("-" * 40)

required_columns = [
    "StaysInWeekendNights",
    "StaysInWeekNights"
]

if all(col in df.columns for col in required_columns):
    total_nights = (
        df["StaysInWeekendNights"] +
        df["StaysInWeekNights"]
    )

    print("Bookings with zero total nights:",
          (total_nights == 0).sum())

    print("Bookings with negative total nights:",
          (total_nights < 0).sum())

# ==================================================
# 9. NUMERICAL SUMMARY
# ==================================================

print("\n8. NUMERICAL SUMMARY")
print("-" * 40)

print(df.describe().T)

# ==================================================
# 10. CHECK COMPLETED
# ==================================================

print("\n" + "=" * 60)
print("DATA CHECK COMPLETED")
print("Original dataset was not modified.")
print("=" * 60)
