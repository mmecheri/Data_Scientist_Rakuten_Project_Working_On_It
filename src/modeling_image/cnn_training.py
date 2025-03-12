import tensorflow as tf

def train_model(model, train_data, val_data, epochs=40, batch_size=64, learning_rate=0.001):
    """
    Entraîne un modèle CNN et retourne l'historique d'entraînement.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size
    )

    return model, history
