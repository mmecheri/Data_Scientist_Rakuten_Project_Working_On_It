import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.label_mapping import  load_label_mapping
from src.display import display_dataframe_to_user  # Ensure this function exists


import config

# Global debug flag
# DEBUG = True  # Set to Ture to enable debug prints
DEBUG = False #Set to False to disable debug prints

def get_classification_df(y_true, y_pred):
    """
    Generate a cleaned classification report DataFrame.
    
    - Converts `classification_report()` to a DataFrame.
    - Removes summary rows (`accuracy`, `macro avg`, `weighted avg`).
    - Converts necessary columns to float (rounded) and integer types.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        classification_df (pd.DataFrame): Cleaned classification report DataFrame.
        summary_rows (pd.DataFrame): Summary rows (accuracy, macro avg, weighted avg).
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True,zero_division=0)
    classification_df = pd.DataFrame(report_dict).T.reset_index()
    classification_df.rename(columns={"index": "Encoded target"}, inplace=True)

    summary_rows = classification_df[classification_df["Encoded target"].isin(["accuracy", "macro avg", "weighted avg"])].copy()
    classification_df = classification_df[~classification_df["Encoded target"].isin(["accuracy", "macro avg", "weighted avg"])]
    
    classification_df["Encoded target"] = classification_df["Encoded target"].astype(int)
    classification_df = classification_df.astype({"precision": float, "recall": float, "f1-score": float, "support": int}).round(2)

    return classification_df, summary_rows

# def load_label_mapping(mapping_path):
#     """
#     Load the label mapping from a pickle file and return it as a DataFrame.
#     """
#     with open(mapping_path, 'rb') as f:
#         prdtypecode_mapping = pickle.load(f)
    
#     mapping_df = pd.DataFrame(prdtypecode_mapping)
#     mapping_df.columns = ["Original prdtypecode", "Encoded target", "Label"]
    
#     return mapping_df

def merge_classification_with_mapping(classification_df, mapping_df):
    """
    Merge the classification report DataFrame with label mapping DataFrame.
    """
    merged_df = classification_df.merge(mapping_df, on="Encoded target", how="left")
    return merged_df[["Encoded target", "Original prdtypecode", "Label", "precision", "recall", "f1-score", "support"]]

def process_classification_report(y_test, y_pred_classes, label_mapping_path):
    """
    Process the classification report:
    - Generates a structured DataFrame
    - Merges with label mapping
    - Formats numerical columns
    - Extracts accuracy and summary rows
    """
    classification_df, summary_rows = get_classification_df(y_test, y_pred_classes)
    mapping_df = load_label_mapping(label_mapping_path)
    classification_df = merge_classification_with_mapping(classification_df, mapping_df)

    classification_df.fillna("", inplace=True)
    
    numeric_columns = ["Encoded target", "Original prdtypecode", "support"]
    for col in numeric_columns:
        classification_df[col] = pd.to_numeric(classification_df[col], errors="coerce").fillna(0).astype(int)
    
    classification_df["Label"] = classification_df["Label"].astype(str)
    classification_df[["precision", "recall", "f1-score"]] = classification_df[["precision", "recall", "f1-score"]].astype(float).round(2)

    accuracy_value = summary_rows.loc[summary_rows["Encoded target"] == "accuracy", "precision"].values[0]
    accuracy_support = classification_df["support"].sum()
    
    accuracy_support = int(accuracy_support)

    summary_rows = summary_rows[summary_rows["Encoded target"].isin(["macro avg", "weighted avg"])].copy()
    summary_rows.rename(columns={"Encoded target": "Metric Type"}, inplace=True)
    summary_rows = summary_rows.round({"precision": 2, "recall": 2, "f1-score": 2})
    summary_rows["support"] = summary_rows["support"].astype(int)

    return classification_df, summary_rows, accuracy_value, accuracy_support

def analyze_classification_performance(
    classification_df, 
    well_classified_threshold=0.80, 
    moderately_classified_threshold=0.50
):
    """
    Analyzes classification performance by categorizing classes based on F1-score.
    
    Args:
        classification_df (pd.DataFrame): Processed classification report DataFrame.
        well_classified_threshold (float, optional): Minimum F1-score for a class to be considered "Well-classified". Defaults to 0.80.
        moderately_classified_threshold (float, optional): Minimum F1-score for a class to be considered "Moderate". Defaults to 0.50.

    Returns:
        category_counts (pd.Series): Count of categories in each classification.
        well_classified (pd.DataFrame): Well-classified categories (F1 ≥ well_classified_threshold).
        moderately_classified (pd.DataFrame): Moderately classified categories (moderately_classified_threshold ≤ F1 < well_classified_threshold).
        poorly_classified (pd.DataFrame): Poorly classified categories (F1 < moderately_classified_threshold).
    """


    classification_df["Quality Classification"] = "Moderate"
    classification_df.loc[classification_df["f1-score"] >= well_classified_threshold, "Quality Classification"] = "Well-classified"
    classification_df.loc[classification_df["f1-score"] < moderately_classified_threshold, "Quality Classification"] = "Poorly classified"

    category_counts = classification_df["Quality Classification"].value_counts()

    poorly_classified = classification_df[classification_df["Quality Classification"] == "Poorly classified"]
    moderately_classified = classification_df[classification_df["Quality Classification"] == "Moderate"]
    well_classified = classification_df[classification_df["Quality Classification"] == "Well-classified"]

    return category_counts, well_classified, moderately_classified, poorly_classified


def generate_confusion_matrix(y_true, y_pred, mapping_path):
    """
    Generates a confusion matrix with mapped labels for better interpretability.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        mapping_path (str): Path to the label mapping pickle file.

    Returns:
        conf_matrix_df (pd.DataFrame): Confusion matrix with mapped labels.
    """
    # Load label mapping
    mapping_df = load_label_mapping(mapping_path)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Get unique class labels (0-26) to use as index and columns
    unique_labels = sorted(set(y_true))  # Ensure sorted order

    # Convert to DataFrame with class indices as rows & columns
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)

    # Map `Encoded target` to meaningful class labels
    class_labels = mapping_df.set_index("Encoded target")["Label"]
    conf_matrix_df = conf_matrix_df.rename(index=class_labels, columns=class_labels)

    return conf_matrix_df



def analyze_misclassifications(conf_matrix_df, top_n=5):
    """
    Identifies the most frequent misclassifications based on the confusion matrix.

    Args:
        conf_matrix_df (pd.DataFrame): Confusion matrix with mapped labels.
        top_n (int): Number of most frequent misclassifications to display.

    Returns:
        misclassified_df (pd.DataFrame): DataFrame listing the most frequent misclassifications.
    """
    misclassifications = []

    # Iterate through the confusion matrix
    for true_label in conf_matrix_df.index:
        for predicted_label in conf_matrix_df.columns:
            if true_label != predicted_label:  # Exclude correct classifications
                count = conf_matrix_df.loc[true_label, predicted_label]
                if count > 0:
                    misclassifications.append((true_label, predicted_label, count))

    # Convert to DataFrame
    misclassified_df = pd.DataFrame(misclassifications, columns=["True Label", "Predicted Label", "Count"])

    # Sort by the highest misclassifications
    misclassified_df = misclassified_df.sort_values(by="Count", ascending=False).head(top_n)

    return misclassified_df

#V2
def analyze_misclassifications_V2(conf_matrix_df, top_n=10):
    """
    Identifies and sorts the most frequent misclassifications.

    Args:
        conf_matrix_df (pd.DataFrame): Confusion matrix with mapped labels.
        top_n (int): Number of most frequent misclassifications to display.

    Returns:
        misclassified_df (pd.DataFrame): DataFrame listing the most frequent misclassifications, sorted by True Label.
    """
    misclassifications = []

    # Iterate through the confusion matrix
    for true_label in conf_matrix_df.index:
        for predicted_label in conf_matrix_df.columns:
            if true_label != predicted_label:  # Exclude correct classifications
                count = conf_matrix_df.loc[true_label, predicted_label]
                if count > 0:
                    misclassifications.append((true_label, predicted_label, count))

    # Convert to DataFrame
    misclassified_df = pd.DataFrame(misclassifications, columns=["True Label", "Predicted Label", "Count"])

    # Sort first by True Label (alphabetical order) and then by Count (descending)
    misclassified_df = misclassified_df.sort_values(by=["True Label", "Count"], ascending=[True, False])

    # Return only the top_n most frequent misclassifications
    return misclassified_df.head(top_n)

def compute_misclassification_rates(conf_matrix_df):
    """
    Computes the misclassification rate for each true class.

    Args:
        conf_matrix_df (pd.DataFrame): Confusion matrix with mapped labels.

    Returns:
        misclassification_rates (pd.DataFrame): DataFrame containing True Label, Support, and Misclassification Rate.
    """
    misclassification_rates = []

    for true_label in conf_matrix_df.index:
        total_actual = conf_matrix_df.loc[true_label].sum()  # Total occurrences of this class
        correct_predictions = conf_matrix_df.loc[true_label, true_label]  # Correct predictions
        misclassified_count = total_actual - correct_predictions  # Incorrect predictions
        
        if total_actual > 0:
            misclassification_rate = misclassified_count / total_actual
            misclassification_rates.append((true_label, total_actual, misclassified_count, misclassification_rate))

    # Convert to DataFrame
    misclassification_rates_df = pd.DataFrame(
        misclassification_rates, 
        columns=["True Label", "Total Samples", "Misclassified Count", "Misclassification Rate"]
    )

    # Sort by misclassification rate in descending order
    misclassification_rates_df = misclassification_rates_df.sort_values(by="Misclassification Rate", ascending=False)

    misclassification_rates_df["Misclassification Rate"] = misclassification_rates_df["Misclassification Rate"].apply(
        lambda x: f"100%" if x == 1 else f"{x*100:.2f}%"
    )

    return misclassification_rates_df

def compute_misclassification_percentage_Old(conf_matrix_df, top_n=10):
    """
    Computes the percentage of misclassification occurrences relative to all misclassified samples.

    Args:
        conf_matrix_df (pd.DataFrame): Confusion matrix with mapped labels.
        top_n (int): Number of most frequent misclassifications to return.

    Returns:
        misclassified_percentage_df (pd.DataFrame): DataFrame with misclassification percentages.
    """
    misclassifications = []

    total_misclassified = conf_matrix_df.values.sum() - np.trace(conf_matrix_df.values)  # Sum of all errors

    for true_label in conf_matrix_df.index:
        for predicted_label in conf_matrix_df.columns:
            if true_label != predicted_label:
                count = conf_matrix_df.loc[true_label, predicted_label]
                if count > 0:
                    percentage = (count / total_misclassified) * 100
                    misclassifications.append((true_label, predicted_label, count, percentage))

    # Convert to DataFrame
    misclassified_percentage_df = pd.DataFrame(
        misclassifications, 
        columns=["True Label", "Predicted Label", "Count", "Percentage"]
    )

    # Sort by percentage
    misclassified_percentage_df = misclassified_percentage_df.sort_values(by="Percentage", ascending=False)

    return misclassified_percentage_df.head(top_n)

def compute_misclassification_percentage(conf_matrix_df, top_n=10):
    """
    Computes the percentage of misclassification occurrences relative to all misclassified samples.

    Args:
        conf_matrix_df (pd.DataFrame): Confusion matrix with mapped labels.
        top_n (int): Number of most frequent misclassifications to return.

    Returns:
        misclassified_percentage_df (pd.DataFrame): DataFrame with misclassification percentages.
    """
    misclassifications = []

    total_misclassified = conf_matrix_df.values.sum() - np.trace(conf_matrix_df.values)  # Total misclassified samples

    for true_label in conf_matrix_df.index:
        for predicted_label in conf_matrix_df.columns:
            if true_label != predicted_label:
                count = conf_matrix_df.loc[true_label, predicted_label]
                if count > 0:
                    percentage = round((count / total_misclassified) * 100, 2)  # Round to 2 decimal places
                    misclassifications.append((true_label, predicted_label, count, f"{percentage}%"))  # Add % symbol

    # Convert to DataFrame
    misclassified_percentage_df = pd.DataFrame(
        misclassifications, 
        columns=["True Label", "Predicted Label", "Count", "Percentage"]
    )

    # Sort by percentage
    misclassified_percentage_df = misclassified_percentage_df.sort_values(by="Percentage", ascending=False)

    return misclassified_percentage_df.head(top_n)

def compute_overpredicted_classes(conf_matrix_df):
    """
    Identifies classes that are most frequently predicted incorrectly.

    Args:
        conf_matrix_df (pd.DataFrame): Confusion matrix with mapped labels.

    Returns:
        overpredicted_df (pd.DataFrame): DataFrame with over-predicted classes.
    """
    overpredictions = []

    for predicted_label in conf_matrix_df.columns:
        total_predicted = conf_matrix_df[predicted_label].sum()  # Total times this class was predicted
        correct_predictions = conf_matrix_df.loc[predicted_label, predicted_label] if predicted_label in conf_matrix_df.index else 0
        incorrect_predictions = total_predicted - correct_predictions
        
        if total_predicted > 0:
            overprediction_rate = round((incorrect_predictions / total_predicted),2)
            # percentage = round((count / total_misclassified) * 100, 2)  # Round to 2 decimal places
            overpredictions.append((predicted_label, total_predicted, incorrect_predictions, overprediction_rate))

    # Convert to DataFrame
    overpredicted_df = pd.DataFrame(
        overpredictions, 
        columns=["Predicted Label", "Total Predicted", "Incorrect Predictions", "Overprediction Rate"]
    )

    # Sort by overprediction rate in descending order
    overpredicted_df = overpredicted_df.sort_values(by="Overprediction Rate", ascending=False)

    overpredicted_df["Overprediction Rate"] = overpredicted_df["Overprediction Rate"].apply(
        lambda x: f"100%" if x == 1 else f"{x*100:.2f}%"
    )

    return overpredicted_df

def analyze_and_display_misclassification(y_test, y_pred, misclassified_counts,
                                            total_samples_per_class,
                                              misclassification_rates,
                                                display_rows=None,
                                                 display_output=True):
    """
        Generates and displays misclassification analysis, including sorting by different metrics.
        
        Args:
            y_test (array-like): True labels from the test set.
            y_pred (array-like): Predicted labels from the model.
            misclassified_counts (pd.Series): Count of misclassified samples per class.
            total_samples_per_class (pd.Series): Total samples per class.
            misclassification_rates (pd.Series): Misclassification rate per class.
            display_rows (int, optional): Number of rows to display. If None, displays the full DataFrame.

        Returns:
            pd.DataFrame: The full misclassification analysis DataFrame.
    """
    # print("\n Generating raw confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)

    # Retrieve the unique encoded labels (classes present in y_test)
    unique_labels = sorted(set(y_test))  

    # Convert the confusion matrix into a DataFrame, keeping Encoded target as index
    conf_matrix_df_raw = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)

    # Load label mapping to retrieve product codes and class labels
    mapping_df = load_label_mapping(config.PRDTYPECODE_MAPPING_PATH)

    # Compute the total number of samples across all classes
    total_samples_all_classes = total_samples_per_class.sum()

    # Create mapping dictionaries
    encoded_to_prdtypecode = mapping_df.set_index("Encoded target")["Original prdtypecode"].to_dict()
    encoded_to_label = mapping_df.set_index("Encoded target")["Label"].to_dict()

    # Construct the misclassification analysis DataFrame
    misclassification_analysis_df = pd.DataFrame({
        "Encoded target": conf_matrix_df_raw.index,  
        "Original prdtypecode": conf_matrix_df_raw.index.map(encoded_to_prdtypecode),
        "Class Label": conf_matrix_df_raw.index.map(encoded_to_label),
        "Samples": total_samples_per_class.values,
        "Misclassified Count": misclassified_counts.values,
        "Misclassification Rate (%)": misclassification_rates.values
    })

    # Compute "Global Misclassification Rate (%)" based on the total dataset size
    misclassification_analysis_df["Global Misclassification Rate (%)"] = (
        (misclassification_analysis_df["Misclassified Count"] / total_samples_all_classes) * 100
    ).round(2)

    print("[✔] Misclassification analysis DataFrame created!")

    if display_output:
        # 1 **Sort by Total Samples** → Understand which classes are most represented  
        misclassification_analysis_df_sorted_samples = misclassification_analysis_df.sort_values(by="Samples", ascending=False)
        display_dataframe_to_user(name="Top classes with the highest number of samples",
                                dataframe=misclassification_analysis_df_sorted_samples, 
                                    display_rows=display_rows)

        # 2 **Sort by Misclassified Count** → Identify which classes have the highest number of errors  
        misclassification_analysis_df_sorted_count = misclassification_analysis_df.sort_values(by="Misclassified Count", ascending=False)
        display_dataframe_to_user(name="Top classes with the highest number of misclassifications",
                                dataframe=misclassification_analysis_df_sorted_count, 
                                    display_rows=display_rows)

        # 3️ **Sort by Misclassification Rate (%)** → Find the classes with the worst performance  
        misclassification_analysis_df_sorted_rate = misclassification_analysis_df.sort_values(by="Misclassification Rate (%)", ascending=False)
        display_dataframe_to_user(name="Top classes with the highest misclassification rate (%)",
                                dataframe=misclassification_analysis_df_sorted_rate, 
                                    display_rows=display_rows)

        # 4️ **Sort by Global Misclassification Rate (%)** → Show which classes impact the overall error rate the most  
        misclassification_analysis_df_sorted_global_rate = misclassification_analysis_df.sort_values(by="Global Misclassification Rate (%)", ascending=False)
        display_dataframe_to_user(name="Top classes contributing the most to overall misclassification (%)",
                                dataframe=misclassification_analysis_df_sorted_global_rate, 
                                    display_rows=display_rows)


    return misclassification_analysis_df  # Return full DataFrame for further analysis

def generate_display_misclass_report(y_true, y_pred, mapping_path, display_rows=None, display_output=True):
    """
        Generates and displays a comprehensive misclassification report combining multiple analyses.

        Args:
            y_true (array-like): True labels from the test set.
            y_pred (array-like): Predicted labels from the model.
            mapping_path (str): Path to the label mapping pickle file.
            display_rows (int, optional): Number of rows to display in the output.
                - If `display_rows` is an integer (e.g., 20), only the top `display_rows` misclassifications are shown.
                - If `display_rows=None`, the full misclassification report is displayed.
                - Default is `None` (showing all rows).

        Returns:
            pd.DataFrame: A DataFrame containing the misclassification report.
                - If `display_rows` is specified, only the top `display_rows` rows are displayed.
                - The full DataFrame is always returned for further analysis.

        Example Usage:
            # Get the full report (all rows)
            full_report = display_misclassification_report(y_test, y_pred_classes, config.PRDTYPECODE_MAPPING_PATH)

            # Get only the top 10 most frequent misclassifications
            top_10_report = display_misclassification_report(y_test, y_pred_classes, config.PRDTYPECODE_MAPPING_PATH, display_rows=10)
        """
    if DEBUG:
        print("\n🔹 Loading label mapping...")
    mapping_df = load_label_mapping(mapping_path)
    if DEBUG:
        print("✅ Label mapping loaded:", mapping_df.shape)

    # Compute confusion matrix
    if DEBUG:
        print("\n🔹 Computing confusion matrix...")
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_labels = mapping_df.set_index("Encoded target")["Label"]

    # Convert to DataFrame with meaningful labels
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
    if DEBUG:
        print("✅ Confusion matrix created:", conf_matrix_df.shape)

    # Compute misclassification counts & rates
    misclassification_data = []
    total_misclassified = conf_matrix_df.values.sum() - np.trace(conf_matrix_df.values)  # Sum of all misclassifications

    for true_label in conf_matrix_df.index:
        total_actual = conf_matrix_df.loc[true_label].sum()
        correct_predictions = conf_matrix_df.loc[true_label, true_label]
        misclassified_count = total_actual - correct_predictions

        if total_actual > 0:
            misclassification_rate = (misclassified_count / total_actual) * 100  # Convert to percentage
        else:
            misclassification_rate = 0.0

        misclassification_data.append((true_label, total_actual, misclassified_count, f"{misclassification_rate:.2f}%"))

    misclassification_rates_df = pd.DataFrame(misclassification_data)

    if misclassification_rates_df.shape[1] == 4:
        misclassification_rates_df.columns = ["True Label", "Class Sample Size", "Misclassified Count", "Misclassification Rate"]
        misclassification_rates_df = misclassification_rates_df.sort_values(by="Misclassified Count", ascending=False)
    else:
        if DEBUG:
            print("⚠️ Issue: Misclassification DataFrame does not have expected columns!")

    if DEBUG:
        print("✅ Misclassification rates DataFrame created:", misclassification_rates_df.shape)

    # Compute frequent misclassifications
    misclassifications = []
    for true_label in conf_matrix_df.index:
        for predicted_label in conf_matrix_df.columns:
            if true_label != predicted_label:
                count = conf_matrix_df.loc[true_label, predicted_label]
                if count > 0:
                    percentage = (count / total_misclassified) * 100
                    misclassifications.append((true_label, predicted_label, count, f"{percentage:.2f}%"))

    # Keep full misclassification DataFrame
    misclassified_percentage_df = pd.DataFrame(
        misclassifications, 
        columns=["True Label", "Predicted Label", "Count", "Percentage"]
    ).sort_values(by="Count", ascending=False)

    if DEBUG:
        print("✅ Misclassified percentage DataFrame created:", misclassified_percentage_df.shape)

    # Compute over-predicted classes
    over_predicted = []
    for predicted_label in conf_matrix_df.columns:
        total_predicted = conf_matrix_df[predicted_label].sum()
        correct_predictions = conf_matrix_df.loc[predicted_label, predicted_label] if predicted_label in conf_matrix_df.index else 0
        incorrect_predictions = total_predicted - correct_predictions

        if total_predicted > 0:
            overprediction_rate = (incorrect_predictions / total_predicted) * 100  # Convert to percentage
        else:
            overprediction_rate = 0.0

        if DEBUG:
            print(f"🔍 Debug: Predicted Label = {predicted_label}, Total Predicted = {total_predicted}, Incorrect Predictions = {incorrect_predictions}, Overprediction Rate = {overprediction_rate:.2f}%")
        
        over_predicted.append((predicted_label, total_predicted, incorrect_predictions, f"{overprediction_rate:.2f}%"))

    over_predicted_df = pd.DataFrame(
        over_predicted, 
        columns=["Predicted Label", "Total Predicted", "Incorrect Predictions", "Overprediction Rate"]
    ).sort_values(by="Overprediction Rate", ascending=False)

    if DEBUG:
        print("✅ Over-predicted classes DataFrame created:", over_predicted_df.shape)

    # 🛠 Debugging: Check for data type mismatches before merging
    if DEBUG:
        print("\n🔍 Debugging Merge Issues:")
        print("True Label dtype in misclassified_percentage_df:", misclassified_percentage_df["True Label"].dtype)
        print("Predicted Label dtype in misclassified_percentage_df:", misclassified_percentage_df["Predicted Label"].dtype)
        print("True Label dtype in misclassification_rates_df:", misclassification_rates_df["True Label"].dtype)
        print("Predicted Label dtype in over_predicted_df:", over_predicted_df["Predicted Label"].dtype)

    # Convert to string for consistent merging
    misclassified_percentage_df["True Label"] = misclassified_percentage_df["True Label"].astype(str)
    misclassified_percentage_df["Predicted Label"] = misclassified_percentage_df["Predicted Label"].astype(str)
    misclassification_rates_df["True Label"] = misclassification_rates_df["True Label"].astype(str)
    over_predicted_df["Predicted Label"] = over_predicted_df["Predicted Label"].astype(str)

    # Merge all dataframes into one enriched report
    consolidated_df = misclassified_percentage_df.merge(
        misclassification_rates_df, on="True Label", how="left"
    ).merge(over_predicted_df, on="Predicted Label", how="left")

    if DEBUG:
        print("\n✅ Consolidated DataFrame created:", consolidated_df.shape)
    
    # Debugging: Print where NaN values appear
    if DEBUG:
        nan_counts = consolidated_df.isna().sum()
        if nan_counts.any():
            print("\n⚠️ Warning: NaN values detected in final DataFrame:")
            print(nan_counts[nan_counts > 0])
        else:
            print("\n✅ No NaN detected.")
    
    if  display_output:
        display_dataframe_to_user( name="Misclassification Report",
                                dataframe=consolidated_df,
                                display_rows=display_rows
       )
    print("\n[✔] Final Consolidated Report Generated!")
    return consolidated_df 