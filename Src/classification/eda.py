"""EDA for the classification task. Run: python -m src.classification.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.data.loader import load_raw, TARGET
from src.classification.features import feature_columns
from src.classification.split import make_split

FIG, TAB = Path("outputs/figures"), Path("outputs/tables")
MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "n_NULL_string": (df == "NULL").sum(),
        "n_unique": df.nunique(),
    })
    summary.to_csv(TAB / "column_summary.csv")
    df.describe().T.round(2).to_csv(TAB / "numeric_describe.csv")
    pd.crosstab(df["ReservationStatus"], df[TARGET]).to_csv(TAB / "leak_reservationstatus.csv")
    pd.crosstab(df["RequiredCarParkingSpaces"], df[TARGET]).to_csv(TAB / "leak_parking.csv")

    cols = feature_columns(df)
    Xtr, Xte, ytr, yte = make_split(df, cols)
    pd.Series({
        "raw_rows": len(df), "exact_duplicate_rows_all31": int(df.duplicated().sum()),
        "rows_after_dedupe(features+target)": len(Xtr) + len(Xte),
        "train": len(Xtr), "test": len(Xte),
        "cancel_rate_raw": round(df[TARGET].mean(), 4),
        "cancel_rate_train": round(ytr.mean(), 4), "cancel_rate_test": round(yte.mean(), 4),
    }).to_csv(TAB / "split_summary.csv", header=["value"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    df[TARGET].value_counts().sort_index().plot.bar(ax=ax[0], color=["#4c72b0", "#c44e52"])
    ax[0].set_title("Class balance (raw)"); ax[0].set_xticklabels(["Not canceled", "Canceled"], rotation=0)
    for lab, g in df.groupby(TARGET):
        ax[1].hist(g["LeadTime"], bins=50, alpha=0.6, label=["Not canceled", "Canceled"][lab])
    ax[1].set_title("LeadTime by outcome"); ax[1].legend(); ax[1].set_xlabel("days")
    fig.tight_layout(); fig.savefig(FIG / "target_and_leadtime.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), ["DepositType", "MarketSegment", "CustomerType", "ArrivalDateMonth"]):
        r = df.groupby(col)[TARGET].mean()
        if col == "ArrivalDateMonth":
            r = r.reindex(MONTHS)
        r.plot.bar(ax=ax, color="#4c72b0"); ax.set_title(f"Cancel rate by {col}"); ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(FIG / "cancel_rate_by_category.png", dpi=120); plt.close(fig)

    # ---- unusual values (raw data, strict feature set columns) ----
    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    pd.Series({
        "Adults == 0": int((df["Adults"] == 0).sum()),
        "No guests at all": int((guests == 0).sum()),
        "Zero nights booked": int((nights == 0).sum()),
        "LeadTime max (days)": int(df["LeadTime"].max()),
        "DaysInWaitingList > 0": int((df["DaysInWaitingList"] > 0).sum()),
        "DaysInWaitingList max": int(df["DaysInWaitingList"].max()),
        "PreviousCancellations > 0": int((df["PreviousCancellations"] > 0).sum()),
        "PreviousCancellations max": int(df["PreviousCancellations"].max()),
    }).to_csv(TAB / "unusual_values.csv", header=["count"])

    # ---- skewness of numeric features used by the classifier ----
    num = df[cols].select_dtypes("number")
    num.skew().round(2).sort_values(ascending=False).to_csv(TAB / "skewness.csv", header=["skew"])

    # ---- skewed features: histograms ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, logy in [(axes[0, 0], "LeadTime", False), (axes[0, 1], "DaysInWaitingList", True),
                          (axes[1, 0], "PreviousCancellations", True), (axes[1, 1], "StaysInWeekNights", True)]:
        df[col].hist(bins=50, ax=ax)
        ax.set_title(col + (" (log y-axis)" if logy else ""))
        if logy:
            ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(FIG / "skewed_features.png", dpi=120); plt.close(fig)

    print(summary); print(pd.read_csv(TAB / "split_summary.csv", index_col=0))
    print("Features used (strict set):", cols)


if __name__ == "__main__":
    main()
