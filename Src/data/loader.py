"""Load and verify H2.csv (City Hotel). Source labels are kept exactly as in Section 4."""
from pathlib import Path
import pandas as pd

SEED = 42
TARGET = "IsCanceled"
DATA_PATH = Path("data/H2.csv")


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert df.shape == (79330, 31), f"Unexpected shape {df.shape}; is this H2.csv?"
    # Text columns carry trailing spaces (e.g. 'BB       ', 'A       '); strip them.
    text_cols = df.select_dtypes(exclude="number").columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())
    return df
