# src/data_preprocessing/load_image_metadata.py

import pandas as pd
from pathlib import Path
import importlib
from IPython.display import display
import config  # your global config file with PROCESSED_DIR and BASE_DIR

# Reload to ensure updates in config are reflected
importlib.reload(config)

def get_relative_path(absolute_path: Path) -> str:
    """
    Returns the relative path from the project root.
    """
    return str(absolute_path.relative_to(config.BASE_DIR))


def load_pickle(file_path: Path, dataset_name: str):
    """
    Loads a pickle file with error handling and preview.
    """
    if not file_path.exists():
        print(f"[X] Error: `{dataset_name}` file not found at {file_path}")
        return None

    try:
        data = pd.read_pickle(file_path)
        print(f"[✔] Loaded `{dataset_name}` | Type: {type(data)}")

        if isinstance(data, pd.DataFrame) and not data.empty:
            display(data.head(1))
        elif isinstance(data, dict) and data:
            sample_items = list(data.items())[:5]
            print(f" Sample entries from `{dataset_name}`: {sample_items}")

        return data
    except Exception as e:
        print(f"[X] Failed to load `{dataset_name}`: {e}")
        return None



def load_image_metadata():
    """
    Loads image metadata files:
    - Train split
    - Validation split
    - Label mapping

    Returns:
    -------
    X_train_df : pd.DataFrame
    X_val_df : pd.DataFrame
    mapping_dict : dict
    """
    processed_dir = Path(config.PROCESSED_DIR)

    files = {
        "X_train_split": processed_dir / "X_train_split.pkl",
        "X_val_split": processed_dir / "X_val_split.pkl",
        "prdtypecode_mapping": processed_dir / "prdtypecode_mapping.pkl"
    }

    # File existence check
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"[X] `{name}` not found at {get_relative_path(path)}")

    # Load files
    X_train_df = load_pickle(files["X_train_split"], "X_train_split")
    X_val_df = load_pickle(files["X_val_split"], "X_val_split")
    mapping_dict = load_pickle(files["prdtypecode_mapping"], "prdtypecode_mapping")

    return X_train_df, X_val_df, mapping_dict
