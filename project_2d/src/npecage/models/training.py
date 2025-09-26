from src.npecage.utils.helpers import f1
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """
    Builds a simplified U-Net model for image segmentation.

    Args:
        IMG_HEIGHT (int): Height of the input images.
        IMG_WIDTH (int): Width of the input images.
        IMG_CHANNELS (int): Number of color channels in the input images.

    Returns:
        keras.Model: Compiled U-Net model ready for training.
    """

    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1])
    model.summary()
    return model


def create_data_generators(patch_dir, patch_size, batch_size=64, seed=42):
    """
    Creates data generators for training and validation datasets using Keras' ImageDataGenerator.

    Args:
        patch_dir (str): Directory path where training and validation image and mask folders are located.
        patch_size (int): Size to which images and masks will be resized.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (train_generator, val_generator, train_image_generator, val_image_generator)
            - train_generator (zip): Zipped generator for training images and masks.
            - val_generator (zip): Zipped generator for validation images and masks.
            - train_image_generator (DirectoryIterator): Generator for training images.
            - val_image_generator (DirectoryIterator): Generator for validation images.
    """
    # Train image generator
    train_image_datagen = ImageDataGenerator(rescale=1. / 255)
    train_image_generator = train_image_datagen.flow_from_directory(
        f'{patch_dir}/train_images',
        target_size=(patch_size, patch_size),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed
    )

    # Train mask generator
    train_mask_datagen = ImageDataGenerator()
    train_mask_generator = train_mask_datagen.flow_from_directory(
        f'{patch_dir}/train_masks',
        target_size=(patch_size, patch_size),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed
    )

    train_generator = zip(train_image_generator, train_mask_generator)

    # Validation image generator
    val_image_datagen = ImageDataGenerator(rescale=1. / 255)
    val_image_generator = val_image_datagen.flow_from_directory(
        f'{patch_dir}/val_images',
        target_size=(patch_size, patch_size),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed
    )

    # Validation mask generator
    val_mask_datagen = ImageDataGenerator()
    val_mask_generator = val_mask_datagen.flow_from_directory(
        f'{patch_dir}/val_masks',
        target_size=(patch_size, patch_size),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed
    )

    val_generator = zip(val_image_generator, val_mask_generator)

    return train_generator, val_generator, train_image_generator, val_image_generator


def train_unet_model(
    patch_dir,
    patch_size=256,
    epochs=100,
    batch_size=64,
    model_fn=simple_unet_model
):
    """
    Trains a U-Net model using the given patch directory and model function.

    Args:
        patch_dir (str): Path to the directory containing training and validation image/mask subfolders.
        patch_size (int, optional): Target patch size for resizing images. Defaults to 256.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        model_fn (function, optional): Function that builds and returns a Keras model. Defaults to simple_unet_model.

    Returns:
        tuple: (model, history)
            - model (keras.Model): Trained U-Net model.
            - history (History): Keras training history object.
    """
    # Create generators
    train_generator, val_generator, train_img_gen, val_img_gen = create_data_generators(
        patch_dir=patch_dir,
        patch_size=patch_size,
        batch_size=batch_size
    )

    steps_per_epoch = train_img_gen.samples // batch_size
    validation_steps = val_img_gen.samples // batch_size

    # Build model
    model = model_fn(patch_size, patch_size, 1)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        mode='min'
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )

    # Fit model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, lr_scheduler]
    )

    return model, history
