import pandas as pd
import pickle

def load_label_mapping(mapping_path):
    """
    Load the label mapping from a pickle file and return it as a DataFrame.
    """
    with open(mapping_path, 'rb') as f:
        prdtypecode_mapping = pickle.load(f)
    
    mapping_df = pd.DataFrame(prdtypecode_mapping)
    mapping_df.columns = ["Original prdtypecode", "Encoded target", "Label"]
    
    return mapping_df