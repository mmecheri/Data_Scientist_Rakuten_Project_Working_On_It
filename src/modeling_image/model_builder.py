from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, Xception, InceptionV3, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(model_name, input_shape=(224, 224, 3), num_classes=27):
    """
    Construit un modèle CNN avec Transfer Learning.
    """
    base_models = {
        "InceptionResNetV2": InceptionResNetV2,
        "DenseNet121": DenseNet121,
        "Xception": Xception,
        "InceptionV3": InceptionV3,
        "MobileNetV2": MobileNetV2
    }
    
    base_model = base_models[model_name](weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False  # On freeze les couches de base

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    return model
