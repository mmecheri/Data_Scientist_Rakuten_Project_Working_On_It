import tensorflow as tf
import os
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import config

is_Debug = True



# CustomModelCheckpoint class definition
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_best_only=False, monitor='val_loss', mode='min', verbose=1):
        super().__init__()
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if self.save_best_only:
            # Check if validation accuracy has improved
            if logs.get('val_accuracy') > self.best_val_accuracy:
                self.best_val_accuracy = logs.get('val_accuracy')
                if self.verbose > 0:
                    relative_path = os.path.relpath(self.filepath, start='.')  # Relative path
                    print(f"Saving model to {relative_path}")
                self.model.save(self.filepath)
        else:
            if self.verbose > 0:
                relative_path = os.path.relpath(self.filepath, start='.')  # Relative path
                print(f"Saving model to {relative_path}")
            self.model.save(self.filepath)
#-----------------------------------------------------------------------------------------------------------------------------------------------

# train_model function definition
def train_model(model, train_data, val_data, epochs=40, batch_size=64, learning_rate=0.001, early_stopping_patience=5, model_name="model_name", save_model_dir=None):
    """
    Trains a CNN model and returns the training history. Saves the model and its checkpoints.

    Args:
    - model: The CNN model to train.
    - train_data: The training data generator.
    - val_data: The validation data generator.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for the optimizer.
    - model_name: The model's name for generating save file names.
    - save_model_dir: The directory where the model and checkpoints should be saved. If None, defaults to `config.IMAGE_MODELS_DIR`.
    - early_stopping_patience: The number of epochs with no improvement after which training will be stopped.

    Returns:
    - model: The trained model.
    - history: The history of the training process.
    """

    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'epochs':<20}: {epochs}")
        print(f"{'batch_size':<20}: {batch_size}")
        print(f"{'learning_rate':<20}: {learning_rate}")
        print(f"{'early_stopping_patience':<20}: {early_stopping_patience}")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'save_model_dir':<20}: {save_model_dir}")
        print(f"{'='*50}\n")

    # If save_model_dir is None, use the default directory from config (e.g., IMAGE_MODELS_DIR)
    if save_model_dir is None:
        save_model_dir = Path(config.IMAGE_MODELS_DIR, model_name)

    save_model_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define the custom checkpoint callback
    checkpoint_callback = CustomModelCheckpoint(
        filepath=save_model_dir / f"checkpoint_{model_name}_{{epoch:02d}}_{{val_accuracy:.2f}}.h5",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # EarlyStopping to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=early_stopping_patience,  # Stop training if validation loss doesn't improve for 5 epochs
        restore_best_weights=True,  # Restore the best weights after stopping
        verbose=1
    )

    # Train the model with the specified callbacks
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping_callback]  # Use the callbacks
    )

    # Save the full model at the end of training
    model.save(save_model_dir / f"{model_name}_model.h5")
    print(f"Model saved at {save_model_dir / f'{model_name}_model.h5'}")

    return model, history


#=================================================================================================================================

def train_model_02(model, train_data, val_data, epochs=40, batch_size=64, learning_rate=0.001, early_stopping_patience=5, model_name="model_name", save_model_dir=None):
    """
    Trains a CNN model and returns the training history. Saves the model and its checkpoints.

    Args:
    - model: The CNN model to train.
    - train_data: The training data generator.
    - val_data: The validation data generator.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for the optimizer.
    - model_name: The model's name for generating save file names.
    - save_model_dir: The directory where the model and checkpoints should be saved. If None, defaults to `config.IMAGE_MODELS_DIR`.
    - early_stopping_patience: The number of epochs with no improvement after which training will be stopped.

    Returns:
    - model: The trained model.
    - history: The history of the training process.
    """

    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'epochs':<20}: {epochs}")
        print(f"{'batch_size':<20}: {batch_size}")
        print(f"{'learning_rate':<20}: {learning_rate}")
        print(f"{'early_stopping_patience':<20}: {early_stopping_patience}")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'save_model_dir':<20}: {save_model_dir}")
        print(f"{'='*50}\n")



    # If save_model_dir is None, use the default directory from config (e.g., IMAGE_MODELS_DIR)
    if save_model_dir is None:
        save_model_dir = Path(config.IMAGE_MODELS_DIR, model_name)

    save_model_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define the checkpoint callback (save the model at each epoch)
    checkpoint_callback = ModelCheckpoint(
        filepath=save_model_dir / f"checkpoint_{model_name}_{{epoch:02d}}_{{val_accuracy:.2f}}.h5",
        monitor='val_accuracy',  # Monitor validation accuracy
        save_best_only=True,  # Save only the best model based on validation accuracy
        save_weights_only=False,  # Save the entire model (not just the weights)
        mode='max',  # Maximizes validation accuracy
        verbose=1
    )

    # EarlyStopping to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=early_stopping_patience,  # Stop training if validation loss doesn't improve for 5 epochs
        restore_best_weights=True,  # Restore the best weights after stopping
        verbose=1
    )

    # Train the model with the specified callbacks
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping_callback]  # Use the callbacks
    )

    # Save the full model at the end of training
    model.save(save_model_dir / f"{model_name}_model.h5")
    print(f"Model saved at {save_model_dir / f'{model_name}_model.h5'}")

    return model, history





def train_model_O1(model, train_data, val_data, epochs=40, batch_size=64, learning_rate=0.001, model_name="model_name"):
    """
    Entraîne un modèle CNN et retourne l'historique d'entraînement.
    
    Args:
    - model: le modèle CNN à entraîner
    - train_data: les données d'entraînement
    - val_data: les données de validation
    - epochs: nombre d'époques pour l'entraînement
    - batch_size: taille des batches d'entraînement
    - learning_rate: taux d'apprentissage
    - model_name: nom du modèle pour générer les noms de fichiers de sauvegarde
    
    Returns:
    - model, history: le modèle entraîné et l'historique d'entraînement
    """
        # Setup paths for saving 
    checkpoints_dir = os.path.join(config.DATA_AUGMENTATION_DIR , model_name, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)  # Crée le répertoire des checkpoints s'il n'existe pas

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    
    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoints_dir, f"checkpoint_{model_name}_{{epoch:02d}}_{{val_accuracy:.2f}}.h5"),
        monitor='val_accuracy',  # Surveille la validation accuracy pour l'arrêt
        save_best_only=True,  # Sauvegarde uniquement le meilleur modèle
        save_weights_only=False,  # Sauvegarde l'ensemble du modèle
        mode='max',  # Maximiser la précision
        verbose=1
    )

    # EarlyStopping to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Surveille la perte de validation
        patience=5,  # Arrêter si la validation ne s'améliore pas pendant 5 époques
        restore_best_weights=True,  # Restaure les meilleurs poids
        verbose=1
    )

    # Entraînement du modèle avec les callbacks
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping_callback]  # Utiliser les callbacks
    )

    return model, history
