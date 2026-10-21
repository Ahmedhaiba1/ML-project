
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "H2.csv"

df = pd.read_csv(DATA_FILE)

# Columns useful for inspecting suspicious bookings
columns = [
    "IsCanceled",
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "ADR",
    "DaysInWaitingList",
    "Country",
    "Agent",
    "Company",
]

# 1. Investigate unusually high ADR
print("\n=== ADR ABOVE 500 ===")
print(df.loc[df["ADR"] > 500, columns].to_string(index=False))

# 2. Investigate unusually high LeadTime
print("\n=== TOP 10 LEADTIME RECORDS ===")
print(
    df.nlargest(10, "LeadTime")[columns]
    .to_string(index=False)
)

# 3. Investigate unusually high number of babies
print("\n=== RECORDS WITH 5 OR MORE BABIES ===")
print(
    df.loc[df["Babies"] >= 5, columns]
    .to_string(index=False)
)

# 4. Investigate long waiting-list records
print("\n=== TOP 10 WAITING-LIST RECORDS ===")
print(
    df.nlargest(10, "DaysInWaitingList")[columns]
    .to_string(index=False)
)

# 5. Check bookings with zero total nights
total_nights = (
    df["StaysInWeekendNights"] + df["StaysInWeekNights"]
)

print("\n=== ZERO-NIGHT BOOKINGS ===")
print("Number of bookings:", (total_nights == 0).sum())

print(
    df.loc[total_nights == 0, columns]
    .head(20)
    .to_string(index=False)
)

# 6. Inspect actual missing values
print("\n=== ROWS WITH MISSING VALUES ===")
print(
    df.loc[df[["Country", "Children"]].isna().any(axis=1),
           columns]
    .to_string(index=False)
)

# 7. Check text placeholders
print("\n=== NULL TEXT PLACEHOLDERS ===")
for col in ["Agent", "Company", "Country"]:
    values = df[col].astype("string").str.strip().str.upper()
    print(f"{col}: {(values == 'NULL').sum()}")

# 8. Check duplicate rows
print("\n=== DUPLICATES ===")
print("Exact duplicate rows after the first occurrence:",
      df.duplicated().sum())
print("Rows belonging to duplicate groups:",
      df.duplicated(keep=False).sum())
