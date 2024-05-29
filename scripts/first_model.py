import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Parameters
path = "firstModelData"  # Folder with all the class folders
labelFile = 'labels.csv'  # File with all names of classes
batch_size_val = 50  # Number of images to process together
steps_per_epoch_val = 2000
epochs_val = 10
image_dimensions = (32, 32)
test_ratio = 0.2  # Percentage of images for testing
validation_ratio = 0.2  # Percentage of remaining images for validation

# Importing the images
def load_images_from_folder(folder_path):
    images = []
    class_no = []
    class_folders = os.listdir(folder_path)
    print(f"Total Classes Detected: {len(class_folders)}")
    for count, class_folder in enumerate(class_folders):
        class_folder_path = os.path.join(folder_path, class_folder)
        image_files = os.listdir(class_folder_path)
        for image_file in image_files:
            image_path = os.path.join(class_folder_path, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_dimensions)
            images.append(img)
            class_no.append(count)
        print(count, end=" ")
    print(" ")
    return np.array(images), np.array(class_no)

images, class_no = load_images_from_folder(path)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, class_no, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)

# Check data shapes
def print_data_shapes():
    print("Data Shapes")
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Validation: {X_validation.shape}, {y_validation.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    assert X_train.shape[0] == y_train.shape[0], "Mismatch in number of training images and labels"
    assert X_validation.shape[0] == y_validation.shape[0], "Mismatch in number of validation images and labels"
    assert X_test.shape[0] == y_test.shape[0], "Mismatch in number of test images and labels"
    assert X_train.shape[1:] == image_dimensions, "Incorrect training image dimensions"
    assert X_validation.shape[1:] == image_dimensions, "Incorrect validation image dimensions"
    assert X_test.shape[1:] == image_dimensions, "Incorrect test image dimensions"

print_data_shapes()

# Read CSV file
data = pd.read_csv(labelFile, encoding='ISO-8859-1')
print(f"Data shape: {data.shape}, {type(data)}")

# Display some sample images of all the classes
def display_samples(data, X_train, y_train, num_classes):
    num_of_samples = []
    cols = 5
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
    fig.tight_layout()
    for i in range(cols):
        for j, row in data.iterrows():
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1)], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(f"{j} - {row['Name']}")
                num_of_samples.append(len(x_selected))
    plt.show()

display_samples(data, X_train, y_train, len(data))

# Display a bar chart showing the number of samples for each category
def display_sample_distribution(num_classes, num_of_samples):
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

num_of_samples = [len(X_train[y_train == i]) for i in range(len(data))]
display_sample_distribution(len(data), num_of_samples)

# Preprocessing the images
def equalize(img):
    return cv2.equalizeHist(img)

def preprocess_images(images):
    images = np.array([equalize(img) for img in images])
    images = images / 255.0
    return images

X_train = preprocess_images(X_train)
X_validation = preprocess_images(X_validation)
X_test = preprocess_images(X_test)

# Add a depth of 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Data augmentation
data_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
data_gen.fit(X_train)

# Show augmented image samples
def show_augmented_images(data_gen, X_train, y_train):
    batches = data_gen.flow(X_train, y_train, batch_size=20)
    X_batch, y_batch = next(batches)
    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()
    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(image_dimensions[0], image_dimensions[1]), cmap=plt.get_cmap("gray"))
        axs[i].axis('off')
    plt.show()

show_augmented_images(data_gen, X_train, y_train)

# Convert labels to categorical
y_train = to_categorical(y_train, len(data))
y_validation = to_categorical(y_validation, len(data))
y_test = to_categorical(y_test, len(data))

# Convolutional Neural Network model
def create_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(image_dimensions[0], image_dimensions[1], 1), activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(data), activation='softmax'))

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Summary of the model
model.summary()

# Training the model
history = model.fit(data_gen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation),
                    shuffle=True)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

