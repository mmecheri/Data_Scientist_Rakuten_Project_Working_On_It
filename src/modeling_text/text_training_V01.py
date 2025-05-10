import tensorflow as tf
import os
import pandas as pd
import pickle 
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np
import config


# is_Debug = True
is_Debug = False

# Assurez-vous que les callbacks sont activés pour calculer F1 score
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        if is_Debug:
            print(f"\n{'='*50}")
            print("F1ScoreCallback - Received arguments:")
  
            print(f"{'='*50}\n")

        self.val_data = val_data


    def on_epoch_end(self, epoch, logs=None):
        # Évaluer le modèle à la fin de chaque époque sur l'ensemble de validation
        y_true = self.val_data.classes
        
        # if is_Debug:
        #   print("\n [INFO] [on_epoch_end] val_f1_score calculation:")
        
        print(f"\nEpoch {epoch + 1}: Evaluating model performance on validation data... Computing weighted F1 score.")        
        
        y_pred = np.argmax(self.model.predict(self.val_data), axis=-1)
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Affichage du F1-score après chaque époque
        if is_Debug:
         print(f"[INFO] [on_epoch_end] Epoch {epoch + 1} - val_f1_score: {f1:.4f}")
        
        if is_Debug:
            print(f"\n{'='*50}")
            print("[INFO] F1ScoreCallback - on_epoch_end - Received arguments:")
            print(f"{'epoch':<20}: {epoch}")
            print(f"{'='*50}\n")


        if is_Debug:
            print(f"[Debug] Checking if 'val_f1_score' exists in logs: {logs.keys()}")

        if 'val_f1_score' not in logs:  # Ajoutez val_f1_score s'il n'existe pas
            logs['val_f1_score'] = f1
            if is_Debug:
                print(f"[Debug] 'val_f1_score' added to logs with value: {f1:.4f}")
        else:
            logs.update({'val_f1_score': f1})  # Sinon, mettez à jour la clé existante
            if is_Debug:
                print(f"[Debug] 'val_f1_score' updated in logs with value: {f1:.4f}")




class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def on_epoch_end(self, epoch, logs=None):
        # Récupérer la valeur du F1-score actuel
        current_f1 = logs.get('val_f1_score')
        
        # Si val_f1_score a effectivement amélioré
        if current_f1 > self.best:
            # Avant de sauvegarder, on récupère le chemin relatif
            relative_path = os.path.relpath(self.filepath, start=os.getcwd())
            print(f"Epoch {epoch + 1}: val_f1_score improved, saving model to {relative_path}")
        
        # Appel de la méthode de la classe parent (qui effectue la sauvegarde du modèle)
        super().on_epoch_end(epoch, logs)

#----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
def train_model_and_save(model, train_data, val_data, epochs=40, batch_size=64, class_weight_dict = None, 
                learning_rate=0.001, early_stopping_patience=5, model_name="model_name", 
                save_model_dir=None, step_name=None, lr_scheduler=None):
    """
        Trains a CNN model and returns the training history. Saves the model and its checkpoints.

        Args:
        - model: The CNN model to train.
        - train_data: The training data generator.
        - val_data: The validation data generator.
        - epochs: Number of epochs for training.
        - batch_size: Batch size for training.
        - class_weight_dict: Dictionary of class weights (optional).
        - learning_rate: Learning rate for the optimizer.
        - model_name: The model's name for generating save file names.
        - save_model_dir: The directory where the model and checkpoints should be saved. If None, defaults to `config.IMAGE_MODELS_DIR`.
        - early_stopping_patience: The number of epochs with no improvement after which training will be stopped.
        - step_name: Represents the current step in the model training strategy (e.g., "Step1", "Step2", etc.) (default is None).
        - lr_scheduler: Learning rate scheduler (default is None, which means no LR scheduler will be applied).
        
        Returns:
        - model: The trained model.
        - history: The history of the training process.
        """

    if is_Debug:
        print(f"\n{'='*50}")
        print("train_model - NEW - Received arguments:")
        print(f"{'epochs':<20}: {epochs}")
        print(f"{'batch_size':<20}: {batch_size}")
            # If class_weight_dict is not None, print the first few class weights or a summary
        if class_weight_dict is not None:
          print(f"{'class_weight_dict (first few items)'}: {list(class_weight_dict.items())[:5]}")
        else :
             print(f"{'class_weight_dict':<20}: {class_weight_dict}")

        print(f"{'learning_rate':<20}: {learning_rate}")
        print(f"{'early_stopping_patience':<20}: {early_stopping_patience}")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'save_model_dir':<20}: {save_model_dir}")
        print(f"{'='*50}\n")
    
    # If save_model_dir is None, use the default directory from config (e.g., IMAGE_MODELS_DIR)
    if save_model_dir is None:
        save_model_dir = Path(config.TEXT_MODELS_DIR, model_name)

    save_model_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callback to stop training if the model no longer improves
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitors for loss of validation
        patience=early_stopping_patience,  # # Stop after early_stopping_patience epochs without improvement
        restore_best_weights=True,  # Restores the best weights
        verbose=1
    )


    # Use step_name to adjust the naming of the checkpoint and model file based on the current training step
    if step_name is not None:
        # Defining file paths with the step_name included for clarity
        checkpoint_filepath = save_model_dir / f"{step_name}_{model_name}_checkpoint.h5"
        model_filepath = save_model_dir / f"{step_name}_{model_name}_model.h5"
        history_filepath = save_model_dir / f"{step_name}_{model_name}_history.pkl"  # File path for saving the training history as .pkl
    else:
        # Defining file paths without step_name for general use
        checkpoint_filepath = save_model_dir / f"{model_name}_checkpoint.h5"
        model_filepath = save_model_dir / f"{model_name}_model.h5"
        history_filepath = save_model_dir / f"{model_name}_history.pkl"  # File path for saving the training history as .pkl
        
    # Define the checkpoint callback (save the model at each epoch)
    checkpoint_callback = CustomModelCheckpoint(
        # filepath=save_model_dir / f"{model_name}checkpoint__{{epoch:02d}}_{{val_f1_score:.2f}}.h5",
        # filepath=save_model_dir / f"{model_name}_checkpoint_val_f1_score-{{val_f1_score:.4f}}.h5",
        filepath=checkpoint_filepath,
        # filepath=save_model_dir / f"checkpoint_{model_name}_{{epoch:02d}}_{{val_accuracy:.2f}}.h5",
        save_best_only=True,
        monitor='val_f1_score',  # Monitore le f1-score au lieu de la val_accuracy
        # monitor='val_accuracy',  # Monitore le f1-score au lieu de la val_accuracy
        mode='max',
        verbose=0
    )

    # Train the model with the specified callbacks
       # If learning rate scheduler is provided, add it to the callbacks list
    callbacks_list = [early_stopping_callback, F1ScoreCallback(val_data=val_data), checkpoint_callback]
    if lr_scheduler is not None:
        callbacks_list = [F1ScoreCallback(val_data=val_data), lr_scheduler]

    # Training
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,  # Apply class weights here
        callbacks=callbacks_list
        # callbacks=[early_stopping_callback, F1ScoreCallback(val_data=val_data),checkpoint_callback]  # Include the F1 callback
    )

    # Save the full model at the end of training
    model.save(model_filepath)
  
    # Save the training history to a .pkl file using pandas to_pickle
    history_df = pd.DataFrame(history.history)  # Convert the dictionary to a pandas DataFrame
    history_df.to_pickle(history_filepath)  # Save the training history DataFrame to the .pkl file

    return model, history

