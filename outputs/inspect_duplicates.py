
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_DIR / "data" / "H2.csv")

# Get all rows that belong to duplicate groups
duplicates = df[df.duplicated(keep=False)].copy()

print("Original rows:", len(df))
print("Rows involved in duplicate groups:", len(duplicates))

# Inspect one repeated group
duplicate_mask = df.duplicated(keep=False)
sample = df.loc[duplicate_mask].head(20)

print("\n=== SAMPLE OF DUPLICATE-GROUP ROWS ===")
print(sample.to_string(index=False))

# Check whether the rows differ in any column
print("\n=== DUPLICATE COUNTS BY KEEP OPTION ===")
print("keep='first':", df.duplicated(keep="first").sum())
print("keep='last':", df.duplicated(keep="last").sum())
print("keep=False:", df.duplicated(keep=False).sum())

# Check if duplicates remain when date/status columns are excluded
columns_without_status = [
    col for col in df.columns
    if col not in ["ReservationStatus", "ReservationStatusDate"]
]

print("\n=== DUPLICATES WITHOUT RESERVATION STATUS COLUMNS ===")
print(
    df.duplicated(subset=columns_without_status).sum()
)
