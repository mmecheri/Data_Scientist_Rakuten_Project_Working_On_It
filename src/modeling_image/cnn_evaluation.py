import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_learning_curve(history, model_name=None):
    """
    Plots training and validation accuracy and loss curves.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy Curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f"Accuracy Curve for {model_name if model_name else 'Model'}")

    # Loss Curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f"Loss Curve for {model_name if model_name else 'Model'}")
    plt.show()

def generate_classification_report(model, test_data, class_names,model_name=None):
    """
    Generates the classification report for the model using test data.
    Args:
        model: The trained model.
        test_data: Data generator or data used for testing.
        class_names: List of class names.
    """
    y_true = test_data.classes
    y_pred = np.argmax(model.predict(test_data), axis=1)

    print(f"Classification Report for {model_name if model_name else 'Model'}:")
    print(classification_report(y_true, y_pred,zero_division=1, target_names=class_names))

def plot_confusion_matrix(model, test_data, class_names,model_name=None):
    """
    Plots a confusion matrix based on predictions from the model.
    """
    y_true = test_data.classes
    y_pred = np.argmax(model.predict(test_data), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name if model_name else 'Model'}")
    plt.show()

def evaluate_model(model, test_data, class_names, history=None,model_name=None, plot_curve=True, plot_cm=True, report=True):
    """
    Evaluates the model using test data. Optionally, plots learning curves, confusion matrix,
    and prints the classification report.

    Args:
        model: The trained model.
        test_data: Data generator or data used for testing.
        class_names: List of class names.
        plot_curve (bool): Whether to plot learning curves (optional).
        plot_cm (bool): Whether to plot confusion matrix (optional).
        report (bool): Whether to print the classification report (optional).
    """
    if model_name is not None:
        print(f"Evaluating model: {model_name}")
    
    if plot_curve and history is not None:
        plot_learning_curve(history, model_name)  # Plot learning curves only if history is provided

    if report:
        generate_classification_report(model, test_data, class_names, model_name)
    
    if plot_cm:
        plot_confusion_matrix(model, test_data, class_names, model_name)


