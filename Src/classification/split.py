"""Deduplicate, then make ONE stratified train/test split. Test set is touched only at the end."""
from sklearn.model_selection import train_test_split
from src.data.loader import SEED, TARGET


def make_split(df, feature_cols, test_size: float = 0.20):
    # Dedupe on (features + target) BEFORE splitting so identical rows cannot land in both sets.
    d = df[feature_cols + [TARGET]].drop_duplicates().reset_index(drop=True)
    X, y = d[feature_cols], d[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
  
