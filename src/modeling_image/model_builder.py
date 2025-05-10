from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, Xception, InceptionV3, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# is_Debug = True
is_Debug = False

def build_model(model_name, input_shape=(224, 224, 3), num_classes=27, unfreeze_last_layers_count=None):
    """
        Builds a CNN model with Transfer Learning, with options to fine-tune the last layers.

        Args:
        - model_name (str): Name of the pre-trained model (e.g., 'InceptionResNetV2').
        - input_shape (tuple): Shape of the input images, default is (224, 224, 3).
        - num_classes (int): Number of output classes for classification.
        - unfreeze_last_layers_count (int, optional): The number of last layers to unfreeze from the base model. 
                                                      If set to a positive integer, the last `unfreeze_last_layers_count` layers will be unfrozen. 
                                                      If None (default), no layers will be unfrozen.

        Returns:
        - model: A TensorFlow Keras Model instance with transfer learning.
    """

    # Check if unfreeze_last_layers_count is a positive integer
    if unfreeze_last_layers_count is not None and (not isinstance(unfreeze_last_layers_count, int) or unfreeze_last_layers_count <= 0):
        raise ValueError("unfreeze_last_layers_count must be a positive integer.")

    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'input_shape':<20}: {input_shape}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'unfreeze_last_layers_count':<20}: {unfreeze_last_layers_count}")
        print(f"{'='*50}\n")

    # Define base models dictionary with corresponding input shapes
    base_models = {
        "InceptionResNetV2": (InceptionResNetV2, (299, 299, 3)),
        "DenseNet121": (DenseNet121, (224, 224, 3)),
        "Xception": (Xception, (299, 299, 3)),
        "InceptionV3": (InceptionV3, (299, 299, 3)),
        "MobileNetV2": (MobileNetV2, (224, 224, 3))
    }
    
    if model_name not in base_models:
        raise ValueError(f"Model name `{model_name}` is not valid. Choose from {list(base_models.keys())}.")
    
    base_model, required_input_shape = base_models[model_name]

    # Ensure the input_shape is correct for the chosen model
    if input_shape != required_input_shape:
        print(f"Warning: The input shape for {model_name} is usually {required_input_shape}. Using {input_shape}.")

    # Build the base model with pre-trained weights
    base_model = base_model(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze all layers of the base model by default
    for layer in base_model.layers:
        layer.trainable = False  # Freeze all layers

    # If unfreeze_last_layers_count is not None, unfreeze the last N layers
    if unfreeze_last_layers_count is not None:
        for layer in base_model.layers[-unfreeze_last_layers_count:]:
            layer.trainable = True

    # Display the frozen/unfrozen status of the last N layers in the base model (debug)
    if is_Debug:
        print(f"\nBase model {model_name}:")
        print(f"unfreeze_last_layers_count {unfreeze_last_layers_count}")
        if unfreeze_last_layers_count is not None :
            for i, layer in enumerate(base_model.layers[-unfreeze_last_layers_count:]):  # Display the last N layers
                print(f"Layer {i + len(base_model.layers)-unfreeze_last_layers_count}: {layer.name} - {'Frozen' if not layer.trainable else 'Unfrozen'}")

    # Add custom classification layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Display the state of added custom layers (all of them)
    if is_Debug:
        print("\nCustom classification layers:")
        for i, layer in enumerate(model.layers[len(base_model.layers):]):
            print(f"Layer {i + len(base_model.layers)}: {layer.name} - {'Trainable' if layer.trainable else 'Frozen'}")

    return model




def build_model_V07(model_name, input_shape=(224, 224, 3), num_classes=27, unfreeze_last_layers_count=None):
    """
        Builds a CNN model with Transfer Learning, with options to fine-tune the last layers.

        Args:
        - model_name (str): Name of the pre-trained model (e.g., 'InceptionResNetV2').
        - input_shape (tuple): Shape of the input images, default is (224, 224, 3).
        - num_classes (int): Number of output classes for classification.
        - unfreeze_last_layers_count (int): The number of last layers to unfreeze from the base model. 
                                            If set to a positive integer, the last `unfreeze_last_layers_count` layers will be unfrozen. 
                                            If None, no layers will be unfrozen.

        Returns:
        - model: A TensorFlow Keras Model instance with transfer learning.
    """


    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'input_shape':<20}: {input_shape}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'unfreeze_last_layers_count':<20}: {unfreeze_last_layers_count}")
        print(f"{'='*50}\n")

    # Define base models dictionary with corresponding input shapes
    base_models = {
        "InceptionResNetV2": (InceptionResNetV2, (299, 299, 3)),
        "DenseNet121": (DenseNet121, (224, 224, 3)),
        "Xception": (Xception, (299, 299, 3)),
        "InceptionV3": (InceptionV3, (299, 299, 3)),
        "MobileNetV2": (MobileNetV2, (224, 224, 3))
    }
    
    if model_name not in base_models:
        raise ValueError(f"Model name `{model_name}` is not valid. Choose from {list(base_models.keys())}.")
    
    base_model, required_input_shape = base_models[model_name]

    # Ensure the input_shape is correct for the chosen model
    if input_shape != required_input_shape:
        print(f"Warning: The input shape for {model_name} is usually {required_input_shape}. Using {input_shape}.")

    # Build the base model with pre-trained weights
    base_model = base_model(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze all layers of the base model by default
    for layer in base_model.layers:
        layer.trainable = False  # Freeze all layers

    # If unfreeze_last_layers_count is not None, unfreeze the last N layers
    if unfreeze_last_layers_count is not None:
        for layer in base_model.layers[-unfreeze_last_layers_count:]:
            layer.trainable = True

    # Display the frozen/unfrozen status of the last 8 layers in the base model (debug)
    if is_Debug:
        print(f"\nBase model {model_name}:")
        for i, layer in enumerate(base_model.layers[-15:]):  # Display the last 8 layers
            print(f"Layer {i + len(base_model.layers)-15}: {layer.name} - {'Frozen' if not layer.trainable else 'Unfrozen'}")

    # Add custom classification layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Display the state of added custom layers (all of them)
    if is_Debug:
        print("\nCustom classification layers:")
        for i, layer in enumerate(model.layers[len(base_model.layers):]):
            print(f"Layer {i + len(base_model.layers)}: {layer.name} - {'Trainable' if layer.trainable else 'Frozen'}")

    return model



#=====================================================================================================================

def build_model_V06(model_name, input_shape=(224, 224, 3), num_classes=27, freeze_base=True, fine_tune_from_layer=None, unfreeze_layers_count=8):
    """
    Builds a CNN model with Transfer Learning, with options to fine-tune certain layers.

    Args:
    - model_name (str): Name of the pre-trained model (e.g., 'InceptionResNetV2').
    - input_shape (tuple): Shape of the input images, default is (224, 224, 3).
    - num_classes (int): Number of output classes for classification.
    - freeze_base (bool): Whether to freeze the base layers or not.
    - fine_tune_from_layer (int): Index of the layer from which to unfreeze (for fine-tuning).
    - unfreeze_layers_count (int): The number of layers from the end to unfreeze.

    Returns:
    - model: A TensorFlow Keras Model instance with transfer learning.
    """
    
    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'input_shape':<20}: {input_shape}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'freeze_base':<20}: {freeze_base}")
        print(f"{'fine_tune_from_layer':<20}: {fine_tune_from_layer}")
        print(f"{'unfreeze_layers_count':<20}: {unfreeze_layers_count}")
        print(f"{'='*50}\n")
    
    # Define base models dictionary with corresponding input shapes
    base_models = {
        "InceptionResNetV2": (InceptionResNetV2, (299, 299, 3)),
        "DenseNet121": (DenseNet121, (224, 224, 3)),
        "Xception": (Xception, (299, 299, 3)),
        "InceptionV3": (InceptionV3, (299, 299, 3)),
        "MobileNetV2": (MobileNetV2, (224, 224, 3))
    }
    
    if model_name not in base_models:
        raise ValueError(f"Model name `{model_name}` is not valid. Choose from {list(base_models.keys())}.")
    
    base_model, required_input_shape = base_models[model_name]

    # Ensure the input_shape is correct for the chosen model
    if input_shape != required_input_shape:
        print(f"Warning: The input shape for {model_name} is usually {required_input_shape}. Using {input_shape}.")

    # Build the base model with pre-trained weights
    base_model = base_model(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze or unfreeze base layers depending on `freeze_base`
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers if freeze_base is True
    else:
        # Fine-tuning layers (use `fine_tune_from_layer` or `unfreeze_layers_count`)
        if fine_tune_from_layer is not None:
            for layer in base_model.layers[:fine_tune_from_layer]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_from_layer:]:
                layer.trainable = True
        elif unfreeze_layers_count > 0:
            # Unfreeze the last `unfreeze_layers_count` layers
            for layer in base_model.layers[:-unfreeze_layers_count]:
                layer.trainable = False
            for layer in base_model.layers[-unfreeze_layers_count:]:
                layer.trainable = True

    # Display the frozen/unfrozen status of the last 8 layers in the base model (debug)
    if is_Debug:
        print(f"\nBase model {model_name}:")
        for i, layer in enumerate(base_model.layers[-8:]):  # Display the last 8 layers
            print(f"Layer {i + len(base_model.layers)-8}: {layer.name} - {'Frozen' if not layer.trainable else 'Unfrozen'}")

    # Add custom classification layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Display the state of added custom layers (all of them)
    if is_Debug:
        print("\nCustom classification layers:")
        for i, layer in enumerate(model.layers[len(base_model.layers):]):
            print(f"Layer {i + len(base_model.layers)}: {layer.name} - {'Trainable' if layer.trainable else 'Frozen'}")

    return model



def build_model_V05(model_name, input_shape=(224, 224, 3), num_classes=27, freeze_base=True, fine_tune_from_layer=None, unfreeze_layers_count=4):
    """
    Builds a CNN model with Transfer Learning, with options to fine-tune certain layers.

    Args:
    - model_name (str): Name of the pre-trained model (e.g., 'InceptionResNetV2').
    - input_shape (tuple): Shape of the input images, default is (224, 224, 3).
    - num_classes (int): Number of output classes for classification.
    - freeze_base (bool): Whether to freeze the base layers or not.
    - fine_tune_from_layer (int): Index of the layer from which to unfreeze (for fine-tuning).
    - unfreeze_layers_count (int): Number of layers to unfreeze for fine-tuning.

    Returns:
    - model: A TensorFlow Keras Model instance with transfer learning.
    """
    
    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'input_shape':<20}: {input_shape}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'freeze_base':<20}: {freeze_base}")
        print(f"{'fine_tune_from_layer':<20}: {fine_tune_from_layer}")
        print(f"{'unfreeze_layers_count':<20}: {unfreeze_layers_count}")
        print(f"{'='*50}\n")
    

    # Define base models dictionary with corresponding input shapes
    base_models = {
        "InceptionResNetV2": (InceptionResNetV2, (299, 299, 3)),
        "DenseNet121": (DenseNet121, (224, 224, 3)),
        "Xception": (Xception, (299, 299, 3)),
        "InceptionV3": (InceptionV3, (299, 299, 3)),
        "MobileNetV2": (MobileNetV2, (224, 224, 3))
    }
    
    if model_name not in base_models:
        raise ValueError(f"Model name `{model_name}` is not valid. Choose from {list(base_models.keys())}.")
    
    base_model, required_input_shape = base_models[model_name]

    # Ensure the input_shape is correct for the chosen model
    if input_shape != required_input_shape:
        print(f"Warning: The input shape for {model_name} is usually {required_input_shape}. Using {input_shape}.")

    # Build the base model with pre-trained weights
    base_model = base_model(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze or unfreeze base layers depending on `freeze_base`
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers if freeze_base is True
    else:
        # Fine-tuning layers
        if fine_tune_from_layer is not None:
            for layer in base_model.layers[:fine_tune_from_layer]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_from_layer:]:
                layer.trainable = True

        # Unfreeze the last `unfreeze_layers_count` layers for fine-tuning
        for layer in base_model.layers[-unfreeze_layers_count:]:
            layer.trainable = True

    # Display the frozen/unfrozen status of the last 8 layers in the base model
    if is_Debug:
        print(f"\nBase model {model_name}:")
        for i, layer in enumerate(base_model.layers[-15:]):  # Only display the last 8 layers
            print(f"Layer {i + len(base_model.layers)-15}: {layer.name} - {'Frozen' if not layer.trainable else 'Unfrozen'}")

    # New Add custom classification layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Display the state of added custom layers (all of them)
    if is_Debug:
        print("\nCustom classification layers:")
        for i, layer in enumerate(model.layers[len(base_model.layers):]):
            print(f"Layer {i + len(base_model.layers)}: {layer.name} - {'Trainable' if layer.trainable else 'Frozen'}")

    return model









##############################################################################################################
def build_model_04(model_name, input_shape=(224, 224, 3), num_classes=27, freeze_base=True, fine_tune_from_layer=None):
    """
    Builds a CNN model with Transfer Learning, with options to fine-tune certain layers.

    Args:
    - model_name (str): Name of the pre-trained model (e.g., 'InceptionResNetV2').
    - input_shape (tuple): Shape of the input images, default is (224, 224, 3).
    - num_classes (int): Number of output classes for classification.
    - freeze_base (bool): Whether to freeze the base layers or not.
    - fine_tune_from_layer (int): Index of the layer from which to unfreeze (for fine-tuning).

    Returns:
    - model: A TensorFlow Keras Model instance with transfer learning.
    """
    
    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'input_shape':<20}: {input_shape}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'freeze_base':<20}: {freeze_base}")
        print(f"{'fine_tune_from_layer':<20}: {fine_tune_from_layer}")
        print(f"{'='*50}\n")
    

    # Define base models dictionary with corresponding input shapes
    base_models = {
        "InceptionResNetV2": (InceptionResNetV2, (299, 299, 3)),
        "DenseNet121": (DenseNet121, (224, 224, 3)),
        "Xception": (Xception, (299, 299, 3)),
        "InceptionV3": (InceptionV3, (299, 299, 3)),
        "MobileNetV2": (MobileNetV2, (224, 224, 3))
    }
    
    if model_name not in base_models:
        raise ValueError(f"Model name `{model_name}` is not valid. Choose from {list(base_models.keys())}.")
    
    base_model, required_input_shape = base_models[model_name]

    # Ensure the input_shape is correct for the chosen model
    if input_shape != required_input_shape:
        print(f"Warning: The input shape for {model_name} is usually {required_input_shape}. Using {input_shape}.")

    # Build the base model with pre-trained weights
    base_model = base_model(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze or unfreeze base layers depending on `freeze_base`
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers if freeze_base is True
    else:
        # Fine-tuning layers
        if fine_tune_from_layer is not None:
            for layer in base_model.layers[:fine_tune_from_layer]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_from_layer:]:
                layer.trainable = True

    # Display the frozen/unfrozen status of the last 8 layers in the base model
    if is_Debug:
        print(f"\nBase model {model_name}:")
        for i, layer in enumerate(base_model.layers[-8:]):  # Only display the last 8 layers
            print(f"Layer {i + len(base_model.layers)-8}: {layer.name} - {'Frozen' if not layer.trainable else 'Unfrozen'}")

    # V OLD -  Add custom classification layers on top of the pre-trained model
    # x = GlobalAveragePooling2D()(base_model.output)
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # output = Dense(num_classes, activation='softmax')(x)

    #V Inhanced?
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Display the state of added custom layers (all of them)
    if is_Debug:
        print("\nCustom classification layers:")
        for i, layer in enumerate(model.layers[len(base_model.layers):]):
            print(f"Layer {i + len(base_model.layers)}: {layer.name} - {'Trainable' if layer.trainable else 'Frozen'}")

    return model


##############################################################################################################
def build_model_V03(model_name, input_shape=(224, 224, 3), num_classes=27, freeze_base=True, fine_tune_from_layer=None):
    """
    Builds a CNN model with Transfer Learning, with options to fine-tune certain layers.

    Args:
    - model_name (str): Name of the pre-trained model (e.g., 'InceptionResNetV2').
    - input_shape (tuple): Shape of the input images, default is (224, 224, 3).
    - num_classes (int): Number of output classes for classification.
    - freeze_base (bool): Whether to freeze the base layers or not.
    - fine_tune_from_layer (int): Index of the layer from which to unfreeze (for fine-tuning).

    Returns:
    - model: A TensorFlow Keras Model instance with transfer learning.
    """
    
    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'input_shape':<20}: {input_shape}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'freeze_base':<20}: {freeze_base}")
        print(f"{'fine_tune_from_layer':<20}: {fine_tune_from_layer}")
        print(f"{'='*50}\n")
    
    # Define base models dictionary with corresponding input shapes
    base_models = {
        "InceptionResNetV2": (InceptionResNetV2, (299, 299)),
        "DenseNet121": (DenseNet121, (224, 224)),
        "Xception": (Xception, (299, 299)),
        "InceptionV3": (InceptionV3, (299, 299)),
        "MobileNetV2": (MobileNetV2, (224, 224))
    }
    
    if model_name not in base_models:
        raise ValueError(f"Model name `{model_name}` is not valid. Choose from {list(base_models.keys())}.")
    
    base_model, required_input_shape = base_models[model_name]

    # Ensure the input_shape is correct for the chosen model
    if input_shape != required_input_shape:
        print(f"Warning: The input shape for {model_name} is usually {required_input_shape}. Using {input_shape}.")

    # Build the base model with pre-trained weights
    base_model = base_model(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze or unfreeze base layers depending on `freeze_base`
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers if freeze_base is True
            if is_Debug:
                print(f"Layer {layer.name} is frozen")
    else:
        # Fine-tuning layers
        if fine_tune_from_layer is not None:
            for layer in base_model.layers[:fine_tune_from_layer]:
                layer.trainable = False
                if is_Debug:
                    print(f"Layer {layer.name} is frozen")
            for layer in base_model.layers[fine_tune_from_layer:]:
                layer.trainable = True
                if is_Debug:
                    print(f"Layer {layer.name} is unfrozen")

    # Add custom classification layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    if is_Debug:
        print(f"Model {model_name} built with input shape {input_shape}. Base model layers {'frozen' if freeze_base else 'unfrozen'}.")
    
    return model








def build_model_V02(model_name, input_shape=(224, 224, 3), num_classes=27, freeze_base=True, fine_tune_from_layer=None):
    """
    Builds a CNN model with Transfer Learning, with options to fine-tune certain layers.

    Args:
    - model_name (str): Name of the pre-trained model (e.g., 'InceptionResNetV2').
    - input_shape (tuple): Shape of the input images, default is (224, 224, 3).
    - num_classes (int): Number of output classes for classification.
    - freeze_base (bool): Whether to freeze the base layers or not.
    - fine_tune_from_layer (int): Index of the layer from which to unfreeze (for fine-tuning).

    Returns:
    - model: A TensorFlow Keras Model instance with transfer learning.
    """
    
    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'input_shape':<20}: {input_shape}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'freeze_base':<20}: {freeze_base}")
        print(f"{'fine_tune_from_layer':<20}: {fine_tune_from_layer}")
        print(f"{'='*50}\n")
    

    # Define base models dictionary with corresponding input shapes
    base_models = {
        "InceptionResNetV2": (InceptionResNetV2, (299, 299)),
        "DenseNet121": (DenseNet121, (224, 224)),
        "Xception": (Xception, (299, 299)),
        "InceptionV3": (InceptionV3, (299, 299)),
        "MobileNetV2": (MobileNetV2, (224, 224))
    }
    
    if model_name not in base_models:
        raise ValueError(f"Model name `{model_name}` is not valid. Choose from {list(base_models.keys())}.")
    
    base_model, required_input_shape = base_models[model_name]

    # Ensure the input_shape is correct for the chosen model
    if input_shape != required_input_shape:
        print(f"Warning: The input shape for {model_name} is usually {required_input_shape}. Using {input_shape}.")

    # Build the base model with pre-trained weights
    base_model = base_model(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze or unfreeze base layers depending on `freeze_base`
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers if freeze_base is True
    else:
        # Fine-tuning layers
        if fine_tune_from_layer is not None:
            for layer in base_model.layers[:fine_tune_from_layer]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_from_layer:]:
                layer.trainable = True

    # Add custom classification layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    if is_Debug:
        print(f"Model {model_name} built with input shape {input_shape}. Base model layers {'frozen' if freeze_base else 'unfrozen'}.")
    
    return model

###########################################################################################################################################
# First Version - TBD
def build_model_V01(model_name, input_shape=(224, 224, 3), num_classes=27, freeze_base=True):
    """
    Builds a CNN model with Transfer Learning.

    Args:
    - model_name (str): Name of the pre-trained model (e.g., 'InceptionResNetV2').
    - input_shape (tuple): Shape of the input images, default is (224, 224, 3).
    - num_classes (int): Number of output classes for classification.
    - freeze_base (bool): Whether to freeze the base layers or not.

    Returns:
    - model: A Keras Model instance with transfer learning.
    """

    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'model_name':<20}: {model_name}")
        print(f"{'input_shape':<20}: {input_shape}")
        print(f"{'num_classes':<20}: {num_classes}")
        print(f"{'freeze_base':<20}: {freeze_base}")
        print(f"{'='*50}\n")
    
    
    # Define base models dictionary with corresponding input shapes
    base_models = {
        "InceptionResNetV2": (InceptionResNetV2, (299, 299 ,3)),
        "DenseNet121": (DenseNet121, (224, 224, 3)),
        "Xception": (Xception, (299, 299, 3)),
        "InceptionV3": (InceptionV3, (299, 299, 3)),
        "MobileNetV2": (MobileNetV2, (224, 224, 3))
    }
    
    if model_name not in base_models:
        raise ValueError(f"Model name `{model_name}` is not valid. Choose from {list(base_models.keys())}.")
    
    base_model, required_input_shape = base_models[model_name]

    # Ensure the input_shape is correct for the chosen model
    if input_shape != required_input_shape:
        print(f"Warning: The input shape for {model_name} is usually {required_input_shape}. Using {input_shape}.")

    # Build the base model with pre-trained weights
    base_model = base_model(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze or unfreeze base layers depending on `freeze_base`
    for layer in base_model.layers:
        layer.trainable = not freeze_base  # Freeze or unfreeze layers based on the parameter

    # Add custom classification layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    if is_Debug:
      print(f"Model {model_name} built with input shape {input_shape}. Base model layers {'frozen' if freeze_base else 'unfrozen'}.")
    
    return model
