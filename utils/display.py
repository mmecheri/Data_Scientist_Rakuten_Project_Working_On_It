from IPython.display import display
import pandas as pd

def display_dataframe_to_user(name: str, dataframe: pd.DataFrame, display_rows=None):
    """
    Display a Pandas DataFrame in a readable format.
    
    Args:
    - name (str): Title of the displayed table.
    - dataframe (pd.DataFrame): The DataFrame to display.
    - display_rows (int, optional): Number of rows to show. If None, displays the full DataFrame.
    """
    print(f"\n{name}: (showing {'all rows' if display_rows is None else f'up to {display_rows} rows'})\n")
    
    # Display entire DataFrame if display_rows is None, otherwise limit to display_rows
    if display_rows is None:
        display(dataframe)  # Show full DataFrame
    else:
        display(dataframe.head(display_rows))  # Show only the top `display_rows`
