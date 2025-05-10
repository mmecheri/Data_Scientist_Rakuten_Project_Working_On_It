# src/data_preprocessing/load_image_metadata.py

import pandas as pd
import numpy as np
from pathlib import Path
import importlib
from IPython.display import display
import config  # your global config file with PROCESSED_DIR and BASE_DIR

# Reload to ensure updates in config are reflected
importlib.reload(config)

is_Debug = False

def get_relative_path(absolute_path: Path) -> str:
    """
    Returns the relative path from the project root.
    """
    return str(absolute_path.relative_to(config.BASE_DIR))


def load_pickle(file_path: Path, dataset_name: str,num_samples = None):
    """
    Loads a pickle file with error handling and preview.
    """
    if not file_path.exists():
        print(f"[X] Error: `{dataset_name}` file not found at {file_path}")
        return None

    try:
        data = pd.read_pickle(file_path)
        if is_Debug:
         print(f"[✔] Loaded `{dataset_name}` | Type: {type(data)}")

        if isinstance(data, pd.DataFrame) and not data.empty:
            if num_samples is not None:
                display(data.head(num_samples))
        elif isinstance(data, dict) and data:
            sample_items = list(data.items())[:5]
            if is_Debug:
             print(f" Sample entries from `{dataset_name}`: {sample_items}")

        return data
    except Exception as e:
        print(f"[X] Failed to load `{dataset_name}`: {e}")
        return None


def load_text_data(num_samples=None):
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
    X_train_df = load_pickle(files["X_train_split"], "X_train_split",num_samples=num_samples)
    X_val_df = load_pickle(files["X_val_split"], "X_val_split",num_samples=num_samples)
    mapping_dict = load_pickle(files["prdtypecode_mapping"], "prdtypecode_mapping",num_samples=num_samples)

    return X_train_df, X_val_df, mapping_dict

def load_tokenized_text_data(num_samples=None):
    """
    Loads tokenized and padded text data:
    - Training data
    - Validation data
    - Trained tokenizer for consistent preprocessing during inference

    Parameters:
    ----------
    num_samples : int, optional (default=None)
        The number of samples to display from the datasets and tokenizer.
        If None, the entire dataset will be displayed.

    Returns:
    -------
    X_train_pad_dl : np.ndarray or pd.DataFrame
    X_val_pad_dl : np.ndarray or pd.DataFrame
    tokenizer_dl : tokenizer object
    """
    processed_dir = Path(config.PROCESSED_DIR)

    files = {
        "X_train_pad_dl": processed_dir / "X_train_pad_dl.pkl",
        "X_val_pad_dl": processed_dir / "X_val_pad_dl.pkl",
        "tokenizer_dl": processed_dir / "tokenizer_dl.pkl"
    }

    # File existence check
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"[X] `{name}` not found at {get_relative_path(path)}")

    # Load files
    X_train_pad_dl = load_pickle(files["X_train_pad_dl"], "X_train_pad_dl")
    X_val_pad_dl = load_pickle(files["X_val_pad_dl"], "X_val_pad_dl")
    tokenizer_dl = load_pickle(files["tokenizer_dl"], "tokenizer_dl")

    # Display a sample of the datasets and tokenizer
    if num_samples is not None:
        # Display 'num_samples' number of rows/tokens if num_samples is provided
        print(f"Sample of X_train_pad_dl (first {num_samples} samples):")
        if isinstance(X_train_pad_dl, np.ndarray):
            print(X_train_pad_dl[:num_samples])  # Display first 'num_samples' rows of the training data
        else:
            print(X_train_pad_dl.head(num_samples))  # If it's a DataFrame, use .head()

        print(f"\nSample of X_val_pad_dl (first {num_samples} samples):")
        if isinstance(X_val_pad_dl, np.ndarray):
            print(X_val_pad_dl[:num_samples])  # Display first 'num_samples' rows of the validation data
        else:
            print(X_val_pad_dl.head(num_samples))  # If it's a DataFrame, use .head()

        print(f"\nTokenizer preview (first {num_samples} tokens):")
        # Display the first 'num_samples' tokens from the tokenizer (example)
        print(list(tokenizer_dl.word_index.items())[:num_samples])

    return X_train_pad_dl, X_val_pad_dl, tokenizer_dl



def load_tfidf_ext_data():
    """
    Loads TF-IDF extracted data:
    - Training data (TF-IDF representation)
    - Validation data (TF-IDF representation)
    - Test data (TF-IDF representation)
    - Trained TF-IDF vectorizer for future transformations

    Returns:
    -------
    X_train_tfidf : scipy.sparse matrix
    X_val_tfidf : scipy.sparse matrix
    tfidf_vectorizer : sklearn TfidfVectorizer object
    """
    processed_dir = Path(config.PROCESSED_DIR)

    files = {
        "X_train_matrix": processed_dir / "Xtrain_matrix.pkl",
        "X_val_matrix": processed_dir / "Xval_matrix.pkl",
        "tfidf_vectorizer": processed_dir / "tfidf_vectorizer.pkl"
    }

    # File existence check
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"[X] `{name}` not found at {get_relative_path(path)}")

    # Load files
    X_train_tfidf = load_pickle(files["X_train_matrix"], "Xtrain_matrix")
    X_val_tfidf = load_pickle(files["X_val_matrix"], "Xval_matrix") 
    tfidf_vectorizer = load_pickle(files["tfidf_vectorizer"], "tfidf_vectorizer")

    return X_train_tfidf, X_val_tfidf, tfidf_vectorizer
