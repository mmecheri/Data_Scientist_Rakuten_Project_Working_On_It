import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score, confusion_matrix

def lr_schedule(epoch, lr):
    """
    Learning rate scheduler that increases the learning rate exponentially.
    The learning rate starts at 1e-5 (0.00001) and increases exponentially
    until reaching approximately 2.54 by the end of the training.
    """
    if epoch == 0:
        print(f"[INFO] : Starting at epoch: {epoch}, learning_rate: {lr}")
        return lr  # Return the learning rate without any change at epoch 0
    else:
        # Update the learning rate exponentially after epoch 0
        lr_updated = 1e-5 * 100 ** (epoch / 37) ## Adjusted to reach ~2.54 at the end of training
        print(f"[INFO] : Learning rate updated from: {lr} to: {lr_updated} at epoch {epoch}")
        return lr_updated





# Function to display Learning Rate across epochs
def display_learning_rate_across_epochs(history, model_name=None):
    """
    Plots the learning rate across epochs.
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configure the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 12)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  #  right border
    
    plt.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', lw=3, linestyle='--', color='black')
    
    plt.title(f'{model_name} - Learning Rate vs Epochs' if model_name else 'Learning Rate vs Epochs', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Learning Rate', size=14)
    plt.legend()
    plt.show()


# Function to display Loss across epochs
def display_loss_across_epochs(history, model_name=None):
    """
    Plots the model's loss across epochs.
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configure the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 12)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  #  right border
    
    plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, linestyle='-', color='blue')
    
    plt.title(f'{model_name} - Loss vs Epochs' if model_name else 'Loss vs Epochs', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Loss', size=14)
    plt.legend()
    plt.show()


# Function to display Accuracy across epochs
def display_accuracy_across_epochs(history, model_name=None):
    """
    Plots the model's accuracy across epochs.
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configure the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 12)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  #  right border
    
    plt.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, linestyle='-', color='purple')
    
    plt.title(f'{model_name} - Accuracy vs Epochs' if model_name else 'Accuracy vs Epochs', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Accuracy', size=14)
    plt.legend()
    plt.show()


# Function to display Validation F1-Score weighted across epochs
def display_val_f1_score_across_epochs(history, model_name=None):
    """
    Plots the model's validation F1-Score across epochs.
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configure the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 12)  # Figure size
    rcParams['axes.spines.top'] = True  # top border
    rcParams['axes.spines.right'] = True  # right border
    
    plt.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1-Score', lw=3, linestyle='-', color='green')
    
    plt.title(f'{model_name} - Validation F1-Score vs Epochs' if model_name else 'Validation F1-Score vs Epochs', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Validation F1-Score', size=14)
    plt.legend()
    plt.show()



def display_lr_vs_val_f1_score(history, model_name=None):
    """
   Validation F1-Score  across   Plots the Learning Rate .
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configuring the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 8)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  #  right border

    # Extract Learning Rate and Validation F1-Score values from the history
    learning_rate = history.history['lr']
    val_f1_score = history.history['val_f1_score']

    # Plotting Learning Rate vs Validation F1-Score
    plt.plot(learning_rate, val_f1_score, label='Validation F1 Score', lw=3)

    # Adding title and labels
    plt.title(f'{model_name} - Learning Rate vs Validation F1-Score' if model_name else 'Learning Rate vs Validation F1-Score', size=20)
    plt.xlabel('Learning Rate', size=14)
    plt.ylabel('Validation F1 Score', size=14)
    plt.legend()

    # Displaying the plot
    plt.show()

def display_lr_loss_accuracy_f1_across_epochs(history, model_name=None):
    """
    Displays Learning Rate, Loss, Accuracy, and F1-Score Weighted across epochs
    with separate y-axes for better visualization and annotations for key points.
    """
    # Configuring the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 14)  # Figure size
    rcParams['axes.spines.top'] = False  # Remove top border
    rcParams['axes.spines.right'] = True  # Show right border
    
    # Creating a figure and axis
    fig, ax1 = plt.subplots()

    # Plotting Loss on the primary y-axis (left axis)
    ax1.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, color='blue')
    ax1.set_xlabel('Epoch', size=14)
    ax1.set_ylabel('Loss', size=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Creating a secondary y-axis (right axis) to plot Accuracy, F1-Score, and Learning Rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, color='purple')
    ax2.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1-Score Weighted', lw=3, color='green')
    ax2.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', lw=3, color='black', linestyle='--')

    ax2.set_ylabel('Accuracy / F1-Score / Learning Rate', size=14)
    ax2.tick_params(axis='y', labelcolor='black')

    # Dynamically find the epoch where F1-Score is the highest
    optimal_epoch = np.argmax(history.history['val_f1_score'])  # Add 1 to match the epoch number


        # Renaming the variables to reflect the changes
    lowest_loss = history.history['loss'][optimal_epoch]  # Lowest Loss value at the optimal epoch
    highest_accuracy = history.history['accuracy'][optimal_epoch]  # Highest Accuracy value at the optimal epoch
    optimal_lr = history.history['lr'][optimal_epoch]  # Learning Rate at the optimal epoch
    highest_f1 = history.history['val_f1_score'][optimal_epoch]  # Highest F1-Score Weighted value at the optimal epoch


    # Adding a vertical line to highlight the optimal LR (at the epoch where F1-Score is highest)
    plt.axvline(x=optimal_epoch, color='red', linestyle=':', label=f'Optimal Epoch at Epoch {optimal_epoch}')

    # Calculate dynamic positioning based on the ranges for each metric
    loss_min, loss_max = min(history.history['loss']), max(history.history['loss'])
    accuracy_min, accuracy_max = min(history.history['accuracy']), max(history.history['accuracy'])
    f1_min, f1_max = min(history.history['val_f1_score']), max(history.history['val_f1_score'])
    lr_min, lr_max = min(history.history['lr']), max(history.history['lr'])

    # Calculate proportional scaling for the annotation positions
    lr_scale = (lr_max - lr_min) / 2  # Scale factor for learning rate
    f1_scale = (f1_max - f1_min) / 2  # Scale factor for F1-score weighted
    accuracy_scale = (accuracy_max - accuracy_min) / 2  # Scale factor for accuracy
    loss_scale = (loss_max - loss_min) / 2  # Scale factor for loss

    print(f"Y-axis range for Loss {min(history.history['loss'])} to {max(history.history['loss'])}")
    print(f"Y-axis range for Loss - loss_scale {loss_scale}")

    # Dynamically set the y-offset for annotations
    lr_offset = lr_scale * 0.8  # Adjust this based on desired space
    f1_offset = f1_scale * 0.8 
    accuracy_offset = accuracy_scale * 0.8 
    loss_offset = loss_scale * 0.5 


    # Annotating the metrics on the plot with corresponding colors
    ax1.annotate(f"Loss: {lowest_loss:.4f}", 
                xy=(optimal_epoch, lowest_loss), 
                xytext=(optimal_epoch + 30, lowest_loss + loss_offset),
                arrowprops=dict(arrowstyle="->", lw=2, color='blue'),  # Arrow color for Loss
                fontsize=12, color='blue')  # Text color for Loss

    ax2.annotate(f"LR: {optimal_lr:.6f}", 
                xy=(optimal_epoch, optimal_lr), 
                xytext=(optimal_epoch + 20, optimal_lr + lr_offset),
                arrowprops=dict(arrowstyle="->", lw=2, color='black'),  # Arrow color for Learning Rate
                fontsize=12, color='black')  # Text color for Learning Rate

    ax2.annotate(f"The highest Weighted F1-Score: {highest_f1:.4f}", 
                xy=(optimal_epoch, highest_f1), 
                xytext=(optimal_epoch + 10, highest_f1 + f1_offset),
                arrowprops=dict(arrowstyle="->", lw=2, color='green'),  # Arrow color for F1-Score
                fontsize=12, color='green')  # Text color for F1-Score

    ax2.annotate(f"Accuracy: {highest_accuracy:.4f}", 
                xy=(optimal_epoch, highest_accuracy), 
                xytext=(optimal_epoch, highest_accuracy + accuracy_offset),
                arrowprops=dict(arrowstyle="->", lw=2, color='purple'),  # Arrow color for Accuracy
                fontsize=12, color='purple')  # Text color for Accuracy



    # Adding the title
    plt.title(f'{model_name} - Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs' if model_name else 'Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs', size=20)

    # Adding Legends with proper locations
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    # Displaying the plot
    plt.show()


def display_lr_loss_accuracy_f1_across_epochs_V08(history, model_name=None):
    """
    Displays Learning Rate, Loss, Accuracy, and F1-Score Weighted across epochs
    with separate y-axes for better visualization and annotations for key points.
    """
    # Configuring the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 14)  # Figure size
    rcParams['axes.spines.top'] = False  # Remove top border
    rcParams['axes.spines.right'] = True  # Show right border
    
    # Creating a figure and axis
    fig, ax1 = plt.subplots()

    # Plotting Loss on the primary y-axis (left axis)
    ax1.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, color='blue')
    ax1.set_xlabel('Epoch', size=14)
    ax1.set_ylabel('Loss', size=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Creating a secondary y-axis (right axis) to plot Accuracy, F1-Score, and Learning Rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, color='purple')
    ax2.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1-Score Weighted', lw=3, color='green')
    ax2.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', lw=3, color='black', linestyle='--')

    ax2.set_ylabel('Accuracy / F1-Score / Learning Rate', size=14)
    ax2.tick_params(axis='y', labelcolor='black')

    # Dynamically find the epoch where F1-Score is the highest
    optimal_lr_epoch = np.argmax(history.history['val_f1_score']) + 1  # Add 1 to match the epoch number

    # Get the learning rate at the optimal epoch
    optimal_lr = history.history['lr'][optimal_lr_epoch - 1]

    # Adding a vertical line to highlight the optimal LR (at the epoch where F1-Score is highest)
    plt.axvline(x=optimal_lr_epoch, color='red', linestyle=':', label=f'Optimal LR at Epoch {optimal_lr_epoch}')

    # Annotating the learning rate value at the optimal epoch
    ax2.annotate(f'LR: {optimal_lr:.5f}', 
                 xy=(optimal_lr_epoch, optimal_lr), 
                 xytext=(optimal_lr_epoch + 2, optimal_lr + 0.1), 
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=12, color='red')

    # Adding the title
    plt.title(f'{model_name} - Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs' if model_name else 'Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs', size=20)

    # Adding Legends with proper locations
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    # Displaying the plot
    plt.show()



def display_lr_loss_accuracy_f1_across_epochs_V07(history, model_name=None):
    """
    Displays Learning Rate, Loss, Accuracy, and F1-Score Weighted across epochs
    with separate y-axes for better visualization and annotations for key points.
    """
    # Configuring the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 14)  # Figure size
    rcParams['axes.spines.top'] = False  # Remove top border
    rcParams['axes.spines.right'] = True  # Show right border
    
    # Creating a figure and axis
    fig, ax1 = plt.subplots()

    # Plotting Loss on the primary y-axis (left axis)
    ax1.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, color='blue')
    ax1.set_xlabel('Epoch', size=14)
    ax1.set_ylabel('Loss', size=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Creating a secondary y-axis (right axis) to plot Accuracy, F1-Score, and Learning Rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, color='purple')
    ax2.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1-Score Weighted', lw=3, color='green')
    ax2.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', lw=3, color='black', linestyle='--')

    ax2.set_ylabel('Accuracy / F1-Score / Learning Rate', size=14)
    ax2.tick_params(axis='y', labelcolor='black')

    # Dynamically find the epoch where F1-Score is the highest
    optimal_lr_epoch = np.argmax(history.history['val_f1_score']) + 1  # Add 1 to match the epoch number
    

    # Adding a vertical line to highlight the optimal LR (at the epoch where F1-Score is highest)
    plt.axvline(x=optimal_lr_epoch, color='red', linestyle=':', label=f'Highest F1-Score at Epoch {optimal_lr_epoch}')

    # Adding the title
    plt.title(f'{model_name} - Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs' if model_name else 'Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs', size=20)

    # Adding Legends with proper locations
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    # Displaying the plot
    plt.show()


def display_lr_loss_accuracy_f1_across_epochs_V05(history, model_name=None):
    """
    Displays Learning Rate, Loss, Accuracy, and F1-Score Weighted across epochs
    with separate y-axes for better visualization and annotations for key points.
    """
    # Configuring the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 14)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  # Show right border
    
    # Creating a figure and axis
    fig, ax1 = plt.subplots()

    # Plotting Loss on the primary y-axis (left axis)
    ax1.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, color='blue')
    ax1.set_xlabel('Epoch', size=14)
    ax1.set_ylabel('Loss', size=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Creating a secondary y-axis (right axis) to plot Accuracy, F1-Score, and Learning Rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, color='purple')
    ax2.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1-Score Weighted', lw=3, color='green')
    ax2.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', lw=3, color='black', linestyle='--')

    ax2.set_ylabel('Accuracy / F1-Score / Learning Rate', size=14)
    ax2.tick_params(axis='y', labelcolor='black')

    # Adding vertical line to highlight the optimal LR (e.g., around 0.0005)
    optimal_lr_epoch = 10  # Suppose optimal LR is at epoch 10
    plt.axvline(x=optimal_lr_epoch, color='red', linestyle=':', label=f'Optimal LR at Epoch {optimal_lr_epoch}')

    # Adding the title
    plt.title(f'{model_name} - Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs' if model_name else 'Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs', size=20)

    # Adding Legends with proper locations
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    # Displaying the plot
    plt.show()



def display_lr_loss_accuracy_f1_across_epochs_V04(history, model_name=None):
    """
    Displays all metrics (Loss, Accuracy, F1-Score Weighted, and Learning Rate) across epochs.
    This function uses two y-axes to ensure that loss, accuracy, and F1-score can be displayed clearly.
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configuring the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 14)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  #  right border
    
    # Creating a figure and axis
    fig, ax1 = plt.subplots()
    
    # Plotting Loss on the primary y-axis (left axis)
    ax1.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, color='blue')
    ax1.set_xlabel('Epoch', size=14)
    ax1.set_ylabel('Loss', size=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Creating a secondary y-axis (right axis) to plot Accuracy, F1-Score, and Learning Rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, color='purple')
    ax2.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1-Score Weighted', lw=3, color='green')
    ax2.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', lw=3, color='black', linestyle='--')

    # Ensuring no negative values for the right y-axis
    ax2.set_ylim(0, max(max(history.history['accuracy']), max(history.history['val_f1_score']), max(history.history['lr'])) * 1.1)
    
    ax2.set_ylabel('Accuracy / F1-Score / Learning Rate', size=14,color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Adding the title
    plt.title(f'{model_name} - Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs' if model_name else 'Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs', size=20)
    
    # Adding Legends
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    
    # Displaying the plot
    plt.show()



def display_lr_loss_accuracy_f1_across_epochs_VO2(history, model_name=None):
    """
    Displays all metrics (Loss, Accuracy, F1-Score Weighted, and Learning Rate) across epochs.
    This function uses two y-axes to ensure that loss, accuracy, and F1-score can be displayed clearly.
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configuring the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 14)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  #  right border
    
    # Creating a figure and axis
    fig, ax1 = plt.subplots()
    
    # Plotting Loss on the primary y-axis (left axis)
    ax1.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, color='blue')
    ax1.set_xlabel('Epoch', size=14)
    ax1.set_ylabel('Loss', size=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Creating a secondary y-axis (right axis) to plot Accuracy, F1-Score, and Learning Rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, color='purple')
    ax2.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1-Score Weighted', lw=3, color='green')
    ax2.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', lw=3, color='black', linestyle='--')
    
    ax2.set_ylabel('Accuracy / F1-Score / Learning Rate', size=14)
    ax2.tick_params(axis='y', labelcolor='black')

    # Adding the title
    plt.title(f'{model_name} - Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs' if model_name else 'Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs', size=20)
    
    # Adding Legends
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    
    # Displaying the plot
    plt.show()



def display_lr_loss_accuracy_f1_across_epochs_V01(history, model_name=None):
    """
    Displays all metrics (Loss, Accuracy, F1-Score Weighted, and Learning Rate) across epochs.
    This function uses two y-axes to ensure that loss, accuracy, and F1-score can be displayed clearly.
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configuring the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 12)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  #  right border
    
    # Creating a figure and axis
    fig, ax1 = plt.subplots()
    
    # Plotting Loss on the primary y-axis (left axis)
    ax1.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, color='blue')
    ax1.set_xlabel('Epoch', size=14)
    ax1.set_ylabel('Loss', size=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Creating a secondary y-axis (right axis) to plot Accuracy, F1-Score, and Learning Rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, color='purple')
    ax2.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1-Score Weighted', lw=3, color='green')
    ax2.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', lw=3, color='red', linestyle='--')
    
    ax2.set_ylabel('Accuracy / F1-Score / Learning Rate', size=14)
    ax2.tick_params(axis='y', labelcolor='black')

    # Adding the title
    plt.title(f'{model_name} - Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs' if model_name else 'Loss, Accuracy, F1-Score Weighted, and Learning Rate vs. Epochs', size=20)
    
    # Adding Legends
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    
    # Displaying the plot
    plt.show()




# Function to display Learning Rate, Loss, Accuracy, and F1-Score Weighted across epochs
def display_lr_loss_accuracy_f1_across_epochs_V0(history, model_name=None):
    """
    Plots the learning rate, loss, accuracy, and F1-Score Weighted across epochs on the same graph.
    
    Args:
    - history: History object returned from model training.
    - model_name (str, optional): Name of the model, used in the plot title.
    """
    # Configure the plot size and aesthetics
    rcParams['figure.figsize'] = (18, 14)  # Figure size
    rcParams['axes.spines.top'] = True  #  top border
    rcParams['axes.spines.right'] = True  #  right border

    # epochs = np.arange(1, len(history.history['loss']) + 1)  

    # Learning Rate
    plt.plot(np.arange(1, len(history.history['lr']) + 1), history.history['lr'], label='Learning Rate', color='black', lw=3, linestyle='--')

    # Loss
    plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Loss', lw=3, color='blue')

    # Accuracy
    plt.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Accuracy', lw=3, color='purple')

    # F1-Score Weighted
    plt.plot(np.arange(1, len(history.history['val_f1_score']) + 1), history.history['val_f1_score'], label='Validation F1 Score', lw=3, color='green')


    # Adding title and labels
    plt.title(f'{model_name} - Learning Rate, Loss, Accuracy, and Validation F1-Score vs Epochs' if model_name else 'Learning Rate, Loss, Accuracy, and Validation F1-Score vs Epochs', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Metrics', size=14)
    plt.legend()
    plt.show()



