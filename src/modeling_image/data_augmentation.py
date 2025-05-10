from tensorflow.keras.preprocessing.image import ImageDataGenerator
import importlib
import config  # your global config file with PROCESSED_DIR and BASE_DIR


# Reload to ensure updates in config are reflected
importlib.reload(config)

is_Debug = True

def create_data_augmentation_generator(input_size, batch_size, train_data, validation_data, augment=True):
    """
    Function to create a data generator with advanced augmentation strategies for image classification.

    Args:
        input_size (tuple): Image size to which the images will be resized (e.g. (224, 224)).
        batch_size (int): Batch size for training and validation.
        train_data (DataFrame): DataFrame containing train data paths and labels.
        validation_data (DataFrame): DataFrame containing validation data paths and labels.
        augment (bool): If True, apply augmentation, else apply only rescaling.

    Returns:
        tuple: train_generator, validation_generator
    """

        # Ensure that labels are in string format (to avoid the TypeError in sparse mode)
    train_data['prdtypecode_encoded'] = train_data['prdtypecode_encoded'].astype(str)
    validation_data['prdtypecode_encoded'] = validation_data['prdtypecode_encoded'].astype(str)
    
    if is_Debug:
        print(f"\n{'='*50}")
        print("Received arguments:")
        print(f"{'input_size':<20}: {input_size}")
        print(f"{'batch_size':<20}: {batch_size}")
        print(f"{'augment':<20}: {augment}")
        print(f"{'train_data_shape':<20}: {train_data.shape}")
        print(f"{'validation_data_shape':<20}: {validation_data.shape}")
        print(f"{'='*50}\n")

    if augment:
        # Data augmentation configuration: applying a range of augmentations to the images
        datagen = ImageDataGenerator(
            rescale=1./255,                    # Rescale pixel values to [0,1]
            rotation_range=30,                 # Random rotations between -30 and 30 degrees
            width_shift_range=0.2,             # Horizontal shift (20% of the image width)
            height_shift_range=0.2,            # Vertical shift (20% of the image height)
            shear_range=0.2,                   # Shear transformation (distort image for variety)
            zoom_range=0.3,                    # Zoom in/out (30% zoom)
            horizontal_flip=True,              # Random horizontal flip
            vertical_flip=True,                # Random vertical flip (to account for different object orientations)
            brightness_range=[0.7, 1.3],      # Adjust brightness randomly between 70% to 130%
            fill_mode='nearest',               # Fill mode for newly created pixels (nearest pixel from neighbors)
            # validation_split=0.2               # Set aside 20% of the data for validation during training
        )
    else:
        # If augmentation is not enabled, only rescale the images
        datagen = ImageDataGenerator(rescale=1./255)

    # Train generator
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=config.RAW_IMAGE_TRAIN_DIR,
        x_col='image_name',
        y_col='prdtypecode_encoded',
        target_size=input_size,
        batch_size=batch_size,
        class_mode='sparse',  # Use sparse because the labels are integers
        # subset='training'
    )

    # Validation generator
    valid_generator = datagen.flow_from_dataframe(
        dataframe=validation_data,
        directory=config.RAW_IMAGE_TRAIN_DIR,
        x_col='image_name',
        y_col='prdtypecode_encoded',
        target_size=input_size,
        batch_size=batch_size,
        class_mode='sparse',  # Use sparse because the labels are integers
        # subset='validation'
    )

    return train_generator, valid_generator





