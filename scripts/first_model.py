import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
path = "myData"
labelFile = 'labels.csv'
batch_size_val = 50
imageDimesions = (32, 32)
testRatio = 0.2
validationRatio = 0.2

# Function to load and preprocess images
def load_images(path):
    images = []
    classNo = []
    count = 0
    myList = os.listdir(path)
    print(f"Total Classes Detected: {len(myList)}")
    print("Importing Classes.....")
    for x in range(len(myList)):
        myPicList = os.listdir(f"{path}/{count}")
        for y in myPicList:
            curImg = cv2.imread(f"{path}/{count}/{y}", cv2.IMREAD_GRAYSCALE)
            curImg = cv2.resize(curImg, imageDimesions)
            images.append(curImg)
            classNo.append(count)
        print(count, end=" ")
        count += 1
    print(" ")
    return np.array(images), np.array(classNo)

# Function to preprocess images
def preprocess_images(images):
    def equalize(img):
        return cv2.equalizeHist(img)
    def normalize(img):
        return img / 255.0
    images = np.array([normalize(equalize(img)) for img in images])
    return images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

# Function to display sample images
def display_sample_images(X_train, y_train, data):
    num_classes = len(np.unique(y_train))
    cols = 5
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
    fig.tight_layout()
    for i in range(cols):
        for j, row in data.iterrows():
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1)], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(f"{j}-{row['Name']}")
    plt.show()

# Function to display data distribution
def display_data_distribution(y_train, num_classes):
    num_of_samples = [len(y_train[y_train == i]) for i in range(num_classes)]
    plt.figure(figsize=(12, 4))
    plt.bar(range(num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

# Function to create the CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, (5, 5), activation='relu', input_shape=input_shape),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main script
if __name__ == "__main__":
    # Load images
    images, classNo = load_images(path)
    print(f"Images shape: {images.shape}")
    print(f"Images type: {images.dtype}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

    # Verify data shapes
    print("Data Shapes")
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Validation: {X_validation.shape}, {y_validation.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    assert X_train.shape[0] == y_train.shape[0]
    assert X_validation.shape[0] == y_validation.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1:] == imageDimesions
    assert X_validation.shape[1:] == imageDimesions
    assert X_test.shape[1:] == imageDimesions

    # Read class labels
    data = pd.read_csv(labelFile, encoding='ISO-8859-1')
    print(f"Data Shape: {data.shape}, {type(data)}")

    # Display sample images
    display_sample_images(X_train, y_train, data)

    # Display data distribution
    display_data_distribution(y_train, len(data))

    # Preprocess images
    X_train = preprocess_images(X_train)
    X_validation = preprocess_images(X_validation)
    X_test = preprocess_images(X_test)

    # Data augmentation
    dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
    dataGen.fit(X_train)

    # Show augmented image samples
    batches = dataGen.flow(X_train, y_train, batch_size=20)
    X_batch, y_batch = next(batches)
    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()
    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]), cmap=plt.get_cmap("gray"))
        axs[i].axis('off')
    plt.show()

    # Convert labels to categorical
    y_train = to_categorical(y_train, len(data))
    y_validation = to_categorical(y_validation, len(data))
    y_test = to_categorical(y_test, len(data))

    # Create model
    model = create_model((imageDimesions[0], imageDimesions[1], 1), len(data))
    print(model.summary())

    # Train model
    history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val), epochs=20, validation_data=(X_validation, y_validation))

    # Plot training history
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    # Evaluate model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Score: {score[0]}')
    print(f'Test Accuracy: {score[1]}')

    # Save model
    model.save("model_kaydedilen.h5")

