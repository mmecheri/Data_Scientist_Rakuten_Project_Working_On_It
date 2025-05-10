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
    def __init__(self, x_val, y_val):
        super().__init__()
        # if is_Debug:
        #     print(f"\n{'='*50}")
        #     print("F1ScoreCallback - Received arguments:")
        #     print(f"{'='*50}\n")

        # x_val et y_val sont directement passés pour évaluer
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        # Utiliser y_val directement pour les vraies étiquettes
        y_true = self.y_val
        
        print(f"\nEpoch {epoch + 1}: Evaluating model performance on validation data... Computing weighted F1 score.")        
        
        # Prédictions du modèle sur les données de validation
        y_pred = np.argmax(self.model.predict(self.x_val), axis=-1)
        
        # Calculer le F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Affichage du F1-score après chaque époque
        if is_Debug:
            print(f"[INFO] [on_epoch_end] Epoch {epoch + 1} - val_f1_score: {f1:.4f}")
        
        if is_Debug:
            print(f"\n{'='*50}")
            print("[INFO] F1ScoreCallback - on_epoch_end - Received arguments:")
            print(f"{'epoch':<20}: {epoch}")
            print(f"{'='*50}\n")

        # Ajouter ou mettre à jour 'val_f1_score' dans les logs
        if 'val_f1_score' not in logs:  # Si 'val_f1_score' n'existe pas dans logs
            logs['val_f1_score'] = f1
            if is_Debug:
                print(f"[Debug] 'val_f1_score' added to logs with value: {f1:.4f}")
        else:
            logs.update({'val_f1_score': f1})  # Sinon, mettre à jour la clé existante
            if is_Debug:
                print(f"[Debug] 'val_f1_score' updated in logs with value: {f1:.4f}")




class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialiser self.best à une très petite valeur, pour éviter 'None'
        # self.best = -np.inf  # Valeur minimale pour commencer les comparaisons

    def on_epoch_end(self, epoch, logs=None):
        # Obtenez le F1 score actuel depuis les logs
        current_f1 = logs.get('val_f1_score')
        
        # if is_Debug:
        #     print(f"\n{'='*50}")
        #     print("current_f1 type")
        #     print(type(current_f1))
        #     print("self.best type")
        #     print(type(self.best))
     
        #     print(f"{'='*50}\n")

        # NEXXX Si le F1 score de validation a amélioré, sauvegardez le modèle
        if current_f1 > self.best:
            self.best = current_f1  # Mettre à jour la meilleure valeur
            relative_path = os.path.relpath(self.filepath, start=os.getcwd())
            print(f"Epoch {epoch + 1}: val_f1_score improved, saving model to {relative_path}")
        
        # Appel à la méthode parent pour sauvegarder le modèle
        super().on_epoch_end(epoch, logs)

# class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#     def on_epoch_end(self, epoch, logs=None):
#         # Get the current F1 score from the logs
#         current_f1 = logs.get('val_f1_score')
        
#         # If the val_f1_score has improved, save the model
#         if current_f1 > self.best:
#             relative_path = os.path.relpath(self.filepath, start=os.getcwd())
#             print(f"Epoch {epoch + 1}: val_f1_score improved, saving model to {relative_path}")
        
#         # Call the parent method to save the model
#         super().on_epoch_end(epoch, logs)


#----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
def train_model_and_save(model, x_train, y_train,x_val,y_val, epochs=40, batch_size=64, 
                class_weight_dict=None, learning_rate=0.001, early_stopping_patience=5, model_name="model_name", 
                save_model_dir=None, step_name=None, lr_scheduler=None):
    """
    Trains the model (Conv1D, DNN, RNN with GRU, RNN with LSTM) and saves the model and its checkpoints.

    Args:
    - model: The model to train (already created model passed as argument).
    - train_data: The training data generator.
    - x_val: The validation data generator.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - class_weight_dict: Dictionary of class weights (optional).
    - learning_rate: Learning rate for the optimizer.
    - model_name: The model's name for generating save file names.
    - save_model_dir: Directory where the model and checkpoints should be saved.
    - early_stopping_patience: Number of epochs with no improvement after which training will stop.
    - lr_scheduler: Learning rate scheduler (optional).
        
    Returns:
    - model: The trained model.
    - history: The training history.
    """

    if is_Debug:
        print(f"\n{'='*50}")
        print(f"train_model - Received arguments:")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        # print(f"Class Weights: {class_weight_dict if class_weight_dict else 'None'}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Early Stopping Patience: {early_stopping_patience}")
        print(f"Model Name: {model_name}")
        print(f"Save Model Directory: {save_model_dir}")
        print(f"{'='*50}\n")
    
    # Set the model saving directory
    if save_model_dir is None:
        save_model_dir = Path(config.TEXT_MODELS_DIR, model_name)

    save_model_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist


    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callback for early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )

    # File paths for saving models and history
    if step_name:
        checkpoint_filepath = save_model_dir / f"{step_name}_{model_name}_checkpoint.h5"
        model_filepath = save_model_dir / f"{step_name}_{model_name}_model.h5"
        history_filepath = save_model_dir / f"{step_name}_{model_name}_history.pkl"
    else:
        checkpoint_filepath = save_model_dir / f"{model_name}_checkpoint.h5"
        model_filepath = save_model_dir / f"{model_name}_model.h5"
        history_filepath = save_model_dir / f"{model_name}_history.pkl"

    checkpoint_callback = CustomModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_f1_score',
        mode='max',
        verbose=0
    )



    callbacks_list = [early_stopping_callback, F1ScoreCallback(x_val=x_val,y_val=y_val), checkpoint_callback]
    if lr_scheduler is not None:
        callbacks_list = [F1ScoreCallback(x_val=x_val,y_val=y_val), lr_scheduler]

    # Training the model
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks_list
    )

    # Save the model and its history
    model.save(model_filepath)
    history_df = pd.DataFrame(history.history)
    history_df.to_pickle(history_filepath)


    return model, history
