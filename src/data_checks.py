
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/H2.csv")

print("=" * 50)
print("1. DUPLICATE ANALYSIS")
print("=" * 50)

print("Exact duplicate rows:", df.duplicated().sum())
print(
    "Rows after removing exact duplicates (count only):",
    len(df.drop_duplicates())
)

print("\n" + "=" * 50)
print("2. ADR ANALYSIS")
print("=" * 50)

print("ADR values equal to zero:", (df["ADR"] == 0).sum())
print("ADR values below zero:", (df["ADR"] < 0).sum())
print("ADR values above 500:", (df["ADR"] > 500).sum())
print("ADR values above 1000:", (df["ADR"] > 1000).sum())

print("\nHighest 10 ADR values:")
print(df["ADR"].nlargest(10).to_list())

print("\n" + "=" * 50)
print("3. MISSING VALUES")
print("=" * 50)

print(df.isnull().sum()[lambda x: x > 0])

print("\n" + "=" * 50)
print("4. POSSIBLE TEXT PLACEHOLDERS")
print("=" * 50)

for col in ["Agent", "Company", "Country", "Meal"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(10))

print("\n" + "=" * 50)
print("5. CANCELLATION DISTRIBUTION")
print("=" * 50)

print(df["IsCanceled"].value_counts())
print("\nCancellation percentages:")
print((df["IsCanceled"].value_counts(normalize=True) * 100).round(2))

print("\n" + "=" * 50)
print("6. STAY LENGTH CHECK")
print("=" * 50)

df["TotalNights"] = (
    df["StaysInWeekendNights"] + df["StaysInWeekNights"]
)

print("Bookings with zero total nights:", (df["TotalNights"] == 0).sum())
print("Bookings with negative total nights:", (df["TotalNights"] < 0).sum())

print("\nChecks completed.")
