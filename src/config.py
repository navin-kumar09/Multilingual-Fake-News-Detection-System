from pathlib import Path

# Paths to datasets and directories
DATA_DIR = Path("dataset")

DATASET_PATHS = {
    "fake_true_csv": {
        "fake": DATA_DIR / "archive (3)" / "fake.csv",
        "true": DATA_DIR / "archive (3)" / "true.csv",
    },
    "ifnd": DATA_DIR / "archive (1)" / "IFND.csv",
    "liar": {
        "train": DATA_DIR / "liar_dataset" / "train.tsv",
        "test": DATA_DIR / "liar_dataset" / "test.tsv",
        "valid": DATA_DIR / "liar_dataset" / "valid.tsv",
    },
    "india_csv": DATA_DIR / "fakenewsincidents_india_dataset_v1" / "fakenewsincidents_india_dataset_v1.csv"
}

# Label Mappings
REAL_LABELS = {"true", "real", "TRUE", "mostly-true", "half-true", "real news", "1"}
FAKE_LABELS = {"fake", "false", "Fake", "FALSE", "barely-true", "pants-fire", "fake news", "0", "FAKE"}

def binarize_label(label):
    import pandas as pd
    if pd.isna(label): 
        return -1
    s = str(label).strip()
    if s in REAL_LABELS: 
        return 1
    if s in FAKE_LABELS: 
        return 0
    s_lower = s.lower()
    if s_lower in {"true", "real", "1", "mostly-true", "half-true"}: 
        return 1
    if s_lower in {"fake", "false", "0", "barely-true", "pants-fire"}: 
        return 0
    return -1
