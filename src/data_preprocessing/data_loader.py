import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image_generators(train_df, valid_df, img_dir, img_size, batch_size, augmentation=True):
    """
    Load train and validation image generators from dataframes.
    
    Args:
    - train_df (pd.DataFrame): Training dataset with image paths and labels.
    - valid_df (pd.DataFrame): Validation dataset with image paths and labels.
    - img_dir (str or Path): Directory containing images.
    - img_size (tuple): Target image size (height, width).
    - batch_size (int): Number of images per batch.
    - augmentation (bool): Whether to apply data augmentation on training data.

    Returns:
    - train_generator (ImageDataGenerator): Generator for training data (with or without augmentation).
    - valid_generator (ImageDataGenerator): Generator for validation data (no augmentation).
    """
    
    # Data augmentation for training (only if augmentation=True)
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            rotation_range=45,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    # No augmentation for validation
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=img_dir,
        x_col="imagePath",
        y_col="prdtypecode",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=img_dir,
        x_col="imagePath",
        y_col="prdtypecode",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False
    )

    return train_generator, valid_generator
