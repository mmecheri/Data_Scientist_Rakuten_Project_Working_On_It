import pandas as pd
import pickle
from IPython.display import display



is_Debug = True  # Vous pouvez définir cette variable sur False pour désactiver les impressions de débogage



def load_label_mapping(mapping_path):
    """
    Load the label mapping from a pickle file and return it as a DataFrame.
    
    Parameters:
    ----------
    mapping_path : str
        The file path of the pickle file containing the label mapping.
    
    Returns:
    -------
    mapping_df : pd.DataFrame or None
        A DataFrame with the label mapping, or None if there was an error.
    """
    try:
        # Load the pickle file directly using pandas
        print(f"[DEBUG] Loading label mapping from: {mapping_path}")
        
        # Read the pickle file into a DataFrame
        mapping_df = pd.read_pickle(mapping_path)
        print(f"[✔] Successfully loaded label mapping from {mapping_path} | Type: {type(mapping_df)}")

        # Check if the DataFrame is valid (not empty)
        if mapping_df.empty:
            print(f"[ERROR] The DataFrame is empty.")
            return None
        
        # Display the first few rows of the DataFrame for debugging
        print(f"[DEBUG] First few rows of the mapping DataFrame:")
        display(mapping_df.head())  # Display the first few rows of the DataFrame

        return mapping_df

    except Exception as e:
        # If an error occurs, display the error message
        print(f"[X] Failed to load label mapping: {e}")
        return None








def load_label_mapping_old(mapping_path):
    """
    Load the label mapping from a pickle file and return it as a DataFrame.
    """
    with open(mapping_path, 'rb') as f:
        prdtypecode_mapping = pickle.load(f)
    
    mapping_df = pd.DataFrame(prdtypecode_mapping)
    mapping_df.columns = ["Original prdtypecode", "Encoded target", "Label"]
    
    return mapping_df