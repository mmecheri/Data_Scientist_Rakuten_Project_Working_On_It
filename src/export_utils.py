
import pandas as pd
import numpy  as np
from pathlib import Path
from datetime import datetime
import config
import xlsxwriter
from pathlib import Path
from datetime import datetime
import config

# #V1 OK
# def export_misclassification_analysis_V0(dataframes_dict, model_name, file_prefix="classification_analysis", mode="text"):
#     """
#     Exports multiple classification-related DataFrames into a single Excel file.

#     Args:
#         dataframes_dict (dict): Dictionary of DataFrames to export, with sheet names as keys.
#         model_name (str): Name of the model used for classification.
#         file_prefix (str, optional): Prefix for the filename. Defaults to "classification_analysis".
#         mode (str, optional): Classification type ("text", "image", "bimodal"). Defaults to "text".

#     Returns:
#         str: The path where the file was saved.
#     """

#     # Select the appropriate save directory based on the classification type
#     if mode == "text":
#         output_dir = Path(config.TEXT_REPORTS_DIR)
#     elif mode == "image":
#         output_dir = Path(config.IMAGE_REPORTS_DIR)
#     elif mode == "bimodal":
#         output_dir = Path(config.BIMODAL_REPORTS_DIR)
#     else:
#         raise ValueError("Invalid mode! Choose from 'text', 'image', or 'bimodal'.")

#     # Ensure the save directory exists
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Generate timestamped filename
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     excel_filename = f"{file_prefix}_{model_name}_{timestamp}.xlsx"
#     excel_path = output_dir / excel_filename

#     # Save all DataFrames in an Excel file with multiple sheets
#     # with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
#     #     for sheet_name, df in dataframes_dict.items():
#     #         df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet names limited to 31 chars
#     with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
#         for sheet_name, df in dataframes_dict.items():
#             # print(f"🛠️ Processing sheet: {sheet_name} | Type: {type(df)}")  # Debugging info

#             if isinstance(df, np.ndarray):  
#                 df = pd.DataFrame(df)  # Convert NumPy arrays to DataFrames

#             elif not isinstance(df, pd.DataFrame):  
#                 raise TypeError(f"[X] Error: Sheet '{sheet_name}' has invalid type: {type(df)}")

#             # Keep index only for confusion matrix to show True Class labels
#             save_index = sheet_name == "Raw Confusion Matrix"

#             df.to_excel(writer, sheet_name=sheet_name[:31], index=save_index)  # Apply index conditionally

#     # print(f"\n[✔] Classification analysis exported successfully to: {excel_path}")
#     return str(excel_path)


# def export_misclassification_analysis_V02(dataframes_dict, model_name, file_prefix="classification_analysis", mode="text"):
#     """
#     Exports multiple classification-related DataFrames into a single Excel file.

#     Args:
#         dataframes_dict (dict): Dictionary of DataFrames to export, with sheet names as keys.
#         model_name (str): Name of the model used for classification.
#         file_prefix (str, optional): Prefix for the filename. Defaults to "classification_analysis".
#         mode (str, optional): Classification type ("text", "image", "bimodal"). Defaults to "text".

#     Returns:
#         str: The path where the file was saved.
#     """

#     # Select the appropriate save directory based on the classification type
#     if mode == "text":
#         output_dir = Path(config.TEXT_REPORTS_DIR)
#     elif mode == "image":
#         output_dir = Path(config.IMAGE_REPORTS_DIR)
#     elif mode == "bimodal":
#         output_dir = Path(config.BIMODAL_REPORTS_DIR)
#     else:
#         raise ValueError("Invalid mode! Choose from 'text', 'image', or 'bimodal'.")

#     # Ensure the save directory exists
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Generate timestamped filename
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     excel_filename = f"{file_prefix}_{model_name}_{timestamp}.xlsx"
#     excel_path = output_dir / excel_filename

#     # Initialize Excel writer
#     with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
#         for sheet_name, df in dataframes_dict.items():
#             if sheet_name == "Class Categorization":  # Special handling for combined export
#                 start_row = 0  # Initialize row position
                
#                 # First, write an empty DataFrame to create the sheet
#                 pd.DataFrame().to_excel(writer, sheet_name=sheet_name)  
                
#                 # Get the existing worksheet from Pandas
#                 worksheet = writer.sheets[sheet_name]

#                 # Get the workbook to apply formatting
#                 workbook = writer.book
#                 bold_format = workbook.add_format({'bold': True, 'font_size': 12, 'align': 'left'})

#                 for category_name, category_df in df.items():
#                     # Write category title (bold format)
#                     worksheet.write(start_row, 0, category_name, bold_format)
#                     start_row += 1  # Move to the next row

#                     # Write DataFrame below the title
#                     category_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=True)
#                     start_row += len(category_df) + 2  # Leave a blank row after each section
            
#             else:
#                 df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Default export behavior

#     return str(excel_path)



# def export_all_analysis_V03(dataframes_dict, model_name, file_prefix="classification_analysis", mode="text"):
#     """
#     Exports multiple classification-related DataFrames into a single Excel file.

#     Args:
#         dataframes_dict (dict): Dictionary of DataFrames to export, with sheet names as keys.
#         model_name (str): Name of the model used for classification.
#         file_prefix (str, optional): Prefix for the filename. Defaults to "classification_analysis".
#         mode (str, optional): Classification type ("text", "image", "bimodal"). Defaults to "text".

#     Returns:
#         str: The path where the file was saved.
#     """

#     # Select the appropriate save directory based on the classification type
#     if mode == "text":
#         output_dir = Path(config.TEXT_REPORTS_DIR)
#     elif mode == "image":
#         output_dir = Path(config.IMAGE_REPORTS_DIR)
#     elif mode == "bimodal":
#         output_dir = Path(config.BIMODAL_REPORTS_DIR)
#     else:
#         raise ValueError("Invalid mode! Choose from 'text', 'image', or 'bimodal'.")

#     # Ensure the save directory exists
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Generate timestamped filename
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     excel_filename = f"{file_prefix}_{model_name}_{timestamp}.xlsx"
#     excel_path = output_dir / excel_filename

#     # Initialize Excel writer
#     with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
#         for sheet_name, df in dataframes_dict.items():

#             # ✅ Special handling for "Class Categorization" (Multiple tables in one sheet)
#             if sheet_name == "Class Categorization":
#                 if not isinstance(df, dict):
#                     raise TypeError(f"[X] Error: Expected a dictionary for '{sheet_name}', but got {type(df)}")

#                 start_row = 0  # Initialize row position
                
#                 # Create an empty DataFrame to initialize the sheet
#                 pd.DataFrame().to_excel(writer, sheet_name=sheet_name)  
                
#                 # Get the worksheet from Pandas
#                 worksheet = writer.sheets[sheet_name]
#                 workbook = writer.book
#                 bold_format = workbook.add_format({'bold': True, 'font_size': 12, 'align': 'left'})

#                 for category_name, category_df in df.items():
#                     # Write category title in bold
#                     worksheet.write(start_row, 0, category_name, bold_format)
#                     start_row += 1  # Move to the next row

#                     # ✅ Ensure DataFrame format before exporting
#                     if isinstance(category_df, np.ndarray):
#                         category_df = pd.DataFrame(category_df)
#                     elif not isinstance(category_df, pd.DataFrame):
#                         raise TypeError(f"[X] Error: Category '{category_name}' in '{sheet_name}' has invalid type: {type(category_df)}")

#                     # Write DataFrame below the title
#                     category_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=True)
#                     start_row += len(category_df) + 2  # Leave a blank row after each section
            
#             else:
#                 # ✅ Convert NumPy arrays to Pandas DataFrames before writing
#                 if isinstance(df, np.ndarray):
#                     df = pd.DataFrame(df)

#                 elif not isinstance(df, pd.DataFrame):  
#                     raise TypeError(f"[X] Error: Sheet '{sheet_name}' has invalid type: {type(df)}")

#                 # ✅ Keep index only for "Raw Confusion Matrix"
#                 save_index = sheet_name == "Raw Confusion Matrix"
#                 df.to_excel(writer, sheet_name=sheet_name[:31], index=save_index)  # Default export behavior

#     return str(excel_path)



def export_all_analysis(dataframes_dict, model_name, file_prefix="classification_analysis", mode="text",
                                      classification_df=None, summary_rows=None, accuracy_value=None, accuracy_support=None):
    """
    Exports multiple classification-related DataFrames into a single Excel file.

    Args:
        dataframes_dict (dict): Dictionary of DataFrames to export, with sheet names as keys.
        model_name (str): Name of the model used for classification.
        file_prefix (str, optional): Prefix for the filename. Defaults to "classification_analysis".
        mode (str, optional): Classification type ("text", "image", "bimodal"). Defaults to "text".
        classification_df (pd.DataFrame, optional): Processed classification report with mapped classes and original labels.
        summary_rows (pd.DataFrame, optional): Summary metrics (macro avg, weighted avg).
        accuracy_value (float, optional): Accuracy value.
        accuracy_support (int, optional): Number of samples used for accuracy calculation.

    Returns:
        str: The path where the file was saved.
    """

    # Select the appropriate save directory based on the classification type
    if mode == "text":
        output_dir = Path(config.TEXT_REPORTS_DIR)
    elif mode == "image":
        output_dir = Path(config.IMAGE_REPORTS_DIR)
    elif mode == "bimodal":
        output_dir = Path(config.BIMODAL_REPORTS_DIR)
    else:
        raise ValueError("Invalid mode! Choose from 'text', 'image', or 'bimodal'.")

    # Ensure the save directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    excel_filename = f"{file_prefix}_{model_name}_{timestamp}.xlsx"
    excel_path = output_dir / excel_filename

    # Initialize Excel writer
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        workbook = writer.book  # Get the workbook object
        
        ### ✅ **1. Export "Processed Classification Report" as the First Sheet**
        if classification_df is not None:
            sheet_name = "Processed Classification Report"
            worksheet = workbook.add_worksheet(sheet_name)  # Create the worksheet
            writer.sheets[sheet_name] = worksheet  # Register it with Pandas writer
            
            bold_format = workbook.add_format({'bold': True, 'font_size': 12, 'align': 'left'})  # Bold formatting
            start_row = 0

            # ✅ Write Accuracy Score
            if accuracy_value is not None and accuracy_support is not None:
                worksheet.write(start_row, 0, f"✔ Accuracy: {accuracy_value:.2f} (on {accuracy_support} samples)", bold_format)
                start_row += 2  # Leave a blank row after

            # ✅ Write Summary Rows
            if summary_rows is not None and not summary_rows.empty:
                worksheet.write(start_row, 0, "✔ Classification Summary Rows", bold_format)
                start_row += 1
                summary_rows.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                start_row += len(summary_rows) + 2  # Leave a blank row

            # ✅ Write Processed Classification Report
            worksheet.write(start_row, 0, "✔ Processed Classification Report with Mapped Classes and Original Labels", bold_format)
            start_row += 1
            classification_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
        
        ### ✅ **2. Export Other DataFrames as Usual**
        for sheet_name, df in dataframes_dict.items():
            if sheet_name == "Class Categorization":
                if not isinstance(df, dict):
                    raise TypeError(f"[X] Error: Expected a dictionary for '{sheet_name}', but got {type(df)}")

                start_row = 0  
                pd.DataFrame().to_excel(writer, sheet_name=sheet_name)  # Create the sheet
                
                worksheet = writer.sheets[sheet_name]
                bold_format = workbook.add_format({'bold': True, 'font_size': 12, 'align': 'left'})

                for category_name, category_df in df.items():
                    worksheet.write(start_row, 0, category_name, bold_format)
                    start_row += 1

                    if isinstance(category_df, np.ndarray):
                        category_df = pd.DataFrame(category_df)
                    elif not isinstance(category_df, pd.DataFrame):
                        raise TypeError(f"[X] Error: Category '{category_name}' in '{sheet_name}' has invalid type: {type(category_df)}")

                    category_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=True)
                    start_row += len(category_df) + 2  
            
            else:
                if isinstance(df, np.ndarray):
                    df = pd.DataFrame(df)
                elif not isinstance(df, pd.DataFrame):  
                    raise TypeError(f"[X] Error: Sheet '{sheet_name}' has invalid type: {type(df)}")

                save_index = sheet_name == "Raw Confusion Matrix"
                df.to_excel(writer, sheet_name=sheet_name[:31], index=save_index)  

    return str(excel_path)
