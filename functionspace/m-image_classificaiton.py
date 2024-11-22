# f-image_classification.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from typing import Union  # Import Union from typing
import shutil
from tensorflow.keras.datasets import cifar10  # Ensure this import is present

# CIFAR-10 Class Names
CIFAR10_CLASSES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

def fetch_cifar10_dataset(
    output_data_path: str = 'dataspace/cifar10_data.npz',
    images_output_dir: str = 'dataspace/cifar10_images',
    train_split: float = 0.8,
    random_seed: int = 42
):
    """
    Fetches the CIFAR-10 dataset, saves it as a NumPy compressed file, and exports images to a directory structure.
    """
    # Load CIFAR-10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Save as NumPy compressed file
    np.savez_compressed(
        output_data_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    print(f"CIFAR-10 dataset saved to {output_data_path}")

    # Create directory structure
    if os.path.exists(images_output_dir):
        shutil.rmtree(images_output_dir)  # Remove existing directory to avoid duplication
    os.makedirs(images_output_dir, exist_ok=True)

    # Function to save images
    def save_images(X, y, subset):
        for idx, (img, label) in enumerate(zip(X, y)):
            class_name = CIFAR10_CLASSES[label[0]]
            class_dir = os.path.join(images_output_dir, subset, class_name)
            os.makedirs(class_dir, exist_ok=True)
            img_path = os.path.join(class_dir, f"{subset}_{idx}.png")
            plt.imsave(img_path, img)

    # Save training images
    save_images(X_train, y_train, 'train')
    print(f"Training images saved to {os.path.join(images_output_dir, 'train')}")

    # Save testing images as validation
    save_images(X_test, y_test, 'validation')
    print(f"Validation images saved to {os.path.join(images_output_dir, 'validation')}")

    return (X_train, y_train), (X_test, y_test)

def train_cnn_model(
    images_dir: str = 'dataspace/cifar10_images',
    model_output_path: str = 'dataspace/cnn_model.h5',
    performance_output_path: str = 'dataspace/cnn_performance.txt',
    epochs: int = 20,
    batch_size: int = 64,
    img_height: int = 32,
    img_width: int = 32
):
    """
    Trains a CNN model using images from a directory structure.
    """
    # Define ImageDataGenerators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches
    train_generator = train_datagen.flow_from_directory(
        os.path.join(images_dir, 'train'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    # Flow validation images in batches
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(images_dir, 'validation'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Save the trained model
    model.save(model_output_path)
    print(f"Trained CNN model saved to {model_output_path}")

    # Save performance metrics
    with open(performance_output_path, 'w') as f:
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        f.write(f"Final Training Accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"Final Training Loss: {final_train_loss:.4f}\n")
        f.write(f"Final Validation Loss: {final_val_loss:.4f}\n")
    print(f"Performance metrics saved to {performance_output_path}")

    return history.history

def cnn_model_predict(
    model_path: str = 'dataspace/cnn_model.h5',
    input_images: Union[list, str] = 'dataspace/real_images/',
    output_data_path: str = 'dataspace/cnn_predictions.csv',
    visualize: bool = False,
    img_height: int = 32,
    img_width: int = 32
):
    """
    Uses the trained CNN model to predict classes of new images.
    """
    model = load_model(model_path)
    predictions = []

    # If input_images is a directory, get all image file paths
    if isinstance(input_images, str) and os.path.isdir(input_images):
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = [
            os.path.join(input_images, fname)
            for fname in os.listdir(input_images)
            if fname.lower().endswith(supported_formats)
        ]
    elif isinstance(input_images, list):
        image_files = input_images
    else:
        raise ValueError("input_images must be a list of file paths or a directory path.")

    if not image_files:
        print("No images found for prediction.")
        return pd.DataFrame()

    for img_path in image_files:
        try:
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 32, 32, 3)
            pred = model.predict(img_array)
            label_idx = np.argmax(pred, axis=1)[0]
            label_name = CIFAR10_CLASSES[label_idx]
            predictions.append({'Image_Path': img_path, 'Predicted_Label': label_name})
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predictions.append({'Image_Path': img_path, 'Predicted_Label': 'Error'})

    df = pd.DataFrame(predictions)
    df.to_csv(output_data_path, index=False)
    print(f"Predictions saved to {output_data_path}")

    # Visualization
    if visualize:
        for _, row in df.iterrows():
            img_path = row['Image_Path']
            label = row['Predicted_Label']
            try:
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.title(f"Predicted: {label}")
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Error displaying {img_path}: {e}")

    return df

def gather_image_paths(directory: str, supported_formats: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')) -> list:
    """
    Gathers all image file paths from the specified directory.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist.")

    image_files = [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.lower().endswith(supported_formats)
    ]

    if not image_files:
        print(f"No images found in directory {directory} with supported formats {supported_formats}.")

    return image_files

