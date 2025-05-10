import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# is_Debug = True
is_Debug = False

# Model architecture selection
def create_model(model_type, vocab_size, embedding_dim, max_sequence_length, num_classes):
    """
    Create and return a model based on the model_type.
    
    Args:
    - model_type (str): Type of model to create (Conv1D, DNN, RNN_GRU, RNN_LSTM).
    - vocab_size (int): Size of the vocabulary (e.g., the maximum number of unique tokens).
    - embedding_dim (int): Dimension of the embedding layer.
    - max_sequence_length (int): Maximum length of input sequences (e.g., number of tokens in each input).
    - num_classes (int): Number of output classes.
    
    Returns:
    - model: The created Keras model.
    """

    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_type':<20}: {model_type}")
        print(f"{'vocab_size':<20}: {vocab_size}")
        print(f"{'embedding_dim':<20}: {embedding_dim}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'='*50}\n")
    
    if model_type == "Conv1D":
            model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
            tf.keras.layers.SpatialDropout1D(0.2),
            tf.keras.layers.Conv1D(64, 2, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif model_type == "DNN":
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, 
                                    output_dim=embedding_dim, 
                                    input_length=max_sequence_length),  # Embedding layer
            tf.keras.layers.Flatten(),  # Flatten the output from the embedding layer
            tf.keras.layers.Dense(100, activation='relu'),  # Dense layer with 100 units
            tf.keras.layers.Dropout(0.5),  # Dropout layer with 0.5 dropout rate
            tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer with softmax
        ])

    elif model_type == "RNN_GRU":
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
            tf.keras.layers.RNN(tf.keras.layers.GRUCell(128), return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    elif model_type == "RNN_LSTM":
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
            tf.keras.layers.RNN(tf.keras.layers.LSTMCell(128), return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
