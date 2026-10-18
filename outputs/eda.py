
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Define file paths
DATA_PATH = Path("data/H2.csv")
OUTPUT_PATH = Path("outputs/eda")

# 2. Create the output folder if it does not exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 3. Load the dataset
df = pd.read_csv(DATA_PATH)

# 4. Display dataset information
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-missing values:")
df.info()

# 5. Check missing values
missing = df.isnull().sum().sort_values(ascending=False)

print("\nMissing values per column:")
print(missing[missing > 0])

# Save missing-value report
missing.to_csv(OUTPUT_PATH / "missing_values.csv")

# 6. Check duplicate rows
duplicates = df.duplicated().sum()
print("\nDuplicate rows:", duplicates)

# 7. Generate descriptive statistics
print("\nNumerical statistics:")
print(df.describe().T)

df.describe(include="all").T.to_csv(
    OUTPUT_PATH / "data_summary.csv"
)

# 8. Check regression target
if "ADR" in df.columns:
    print("\nADR statistics:")
    print(df["ADR"].describe())

    df["ADR"].hist(bins=40)
    plt.title("ADR Distribution")
    plt.xlabel("ADR")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "adr_distribution.png")
    plt.close()

# 9. Check classification target
if "IsCanceled" in df.columns:
    print("\nCancellation distribution:")
    print(df["IsCanceled"].value_counts(dropna=False))

    df["IsCanceled"].value_counts().sort_index().plot(
        kind="bar"
    )
    plt.title("Booking Cancellation Distribution")
    plt.xlabel("IsCanceled (0 = Not canceled, 1 = Canceled)")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_PATH / "cancellation_distribution.png"
    )
    plt.close()

print("\nEDA completed successfully!")
print("Reports and plots saved in outputs/eda/")
