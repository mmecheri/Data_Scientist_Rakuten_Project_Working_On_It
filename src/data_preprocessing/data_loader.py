# src/data_preprocessing/load_image_metadata.py

import pandas as pd
import numpy as np
from pathlib import Path
import importlib
from IPython.display import display
import config  # your global config file with PROCESSED_DIR and BASE_DIR

# Reload to ensure updates in config are reflected
importlib.reload(config)
is_Debug = True

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
        print(f"[✔] Loaded `{dataset_name}` | Type: {type(data)}")

        if isinstance(data, pd.DataFrame) and not data.empty:
            if num_samples is not None:
                display(data.head(num_samples))
        elif isinstance(data, dict) and data:
            sample_items = list(data.items())[:5]
            print(f" Sample entries from `{dataset_name}`: {sample_items}")

        return data
    except Exception as e:
        print(f"[X] Failed to load `{dataset_name}`: {e}")
        return None


def load_splitted_data(num_samples=None):
    """
    Loads  files:
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


def load_tfidf_ext_data(num_samples=None):
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

        # Display a sample of the datasets and tokenizer
    if num_samples is not None:
        # Display 'num_samples' number of rows/tokens if num_samples is provided
        print(f"Sample of X_train_tfidf (first {num_samples} samples):")
        # print(X_train_tfidf[:num_samples])
        print(pd.DataFrame(X_train_tfidf[:num_samples].todense()).head())

        print(f"\nSample of X_val_tfidf (first {num_samples} samples):")
        # print(X_val_tfidf[:num_samples])
        print(pd.DataFrame(X_val_tfidf[:num_samples].todense()).head())

        print(f"\tfidf_vectorizer preview (first {num_samples} tokens):")
        # Display the first 'num_samples' tokens from the tokenizer (example)
        print(list(tfidf_vectorizer.vocabulary_.items())[:num_samples])

    return X_train_tfidf, X_val_tfidf, tfidf_vectorizer




def load_product_code_mapping(num_samples=None):
    """
    Loads product code mapping data:
    - prdtypecode_mapping.pkl

    Returns:
    -------
    mapping_df : pd.DataFrame
    """
    processed_dir = Path(config.PROCESSED_DIR)

    files = {
        "prdtypecode_mapping": processed_dir / "prdtypecode_mapping.pkl"
    }

    # File existence check
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"[X] `{name}` not found at {get_relative_path(path)}")

    # Load the mapping data
    mapping_df = load_pickle(files["prdtypecode_mapping"], "prdtypecode_mapping", num_samples=num_samples)

    return mapping_df



def load_submission_data(num_samples=None):
    """
    Loads submission data:
    - Test data (X_test_sub_cleaned_final.pkl)

    Returns:
    -------
    X_test_df : pd.DataFrame
    """
    processed_dir = Path(config.PROCESSED_DIR)

    files = {
        "X_test_sub_cleaned_final": processed_dir / "X_test_sub_cleaned_final.pkl"
    }

    # File existence check
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"[X] `{name}` not found at {get_relative_path(path)}")

    # Load the test data
    X_test_df = load_pickle(files["X_test_sub_cleaned_final"], "X_test_sub_cleaned_final", num_samples=num_samples)

    return X_test_df




def map_encoded_predictions_to_labels(predictions, mapping_df):
    """
    Maps encoded predictions (0-26) to the original prdtypecode and human-readable Label.

    Parameters
    ----------
    predictions : array-like or pd.Series
        Encoded predicted class labels (e.g., values between 0 and 26).

    mapping_df : pd.DataFrame
        DataFrame containing columns: ["Encoded target", "Original prdtypecode", "Label"].

    Returns
    -------
    mapped_df : pd.DataFrame
        DataFrame with columns: ["Predicted Encoded", "prdtypecode", "Label"].
    """
    # Ensure correct types for matching
    predictions_series = pd.Series(predictions).astype(int)
    mapping_df = mapping_df.copy()
    mapping_df['Encoded target'] = mapping_df['Encoded target'].astype(int)

    if is_Debug:
        print("[DEBUG] Unique predictions:", sorted(predictions_series.unique()))
        print("[DEBUG]  Encoded target values available in mapping table:", sorted(mapping_df['Encoded target'].unique()))

    # Build mapping dictionary
    mapping_dict = mapping_df.set_index('Encoded target')[['Original prdtypecode', 'Label']].to_dict(orient='index')

    # Apply mapping
    mapped = predictions_series.map(lambda x: mapping_dict.get(x, {'Original prdtypecode': None, 'Label': None}))

    # Build final DataFrame
    mapped_df = pd.DataFrame(mapped.tolist())
    mapped_df['Predicted Encoded'] = predictions_series
    mapped_df.rename(columns={'Original prdtypecode': 'prdtypecode'}, inplace=True)

    # Reorder columns
    mapped_df = mapped_df[['Predicted Encoded', 'prdtypecode', 'Label']]

    return mapped_df


def map_encoded_predictions_to_labels_V1(predictions, mapping_df):
    """
    Maps encoded predictions (0-26) to the original prdtypecode and human-readable Label.

    Parameters
    ----------
    predictions : array-like or pd.Series
        Encoded predicted class labels (e.g., values between 0 and 26).

    mapping_df : pd.DataFrame
        DataFrame containing columns: ["Encoded target", "Original prdtypecode", "Label"].

    Returns
    -------
    mapped_df : pd.DataFrame
        DataFrame with columns: ["Predicted Encoded", "prdtypecode", "Label"].
    """


    # Create mapping dictionary from encoded class to original prdtypecode and label
    mapping_dict = mapping_df.set_index('Encoded target')[['Original prdtypecode', 'Label']].to_dict(orient='index')

    # Convert predictions to pandas Series if not already
    predictions_series = pd.Series(predictions)

    # Apply mapping
    mapped = predictions_series.map(lambda x: mapping_dict.get(x))

    # Build resulting DataFrame
    mapped_df = pd.DataFrame(mapped.tolist(), columns=["prdtypecode", "Label"])
    mapped_df["Predicted Encoded"] = predictions_series

    # Reorder columns
    mapped_df = mapped_df[["Predicted Encoded", "prdtypecode", "Label"]]

    return mapped_df
