import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

################# Parameters #####################

path = "firstModelData"  # folder with all the class folders
label_file = 'labels.csv'  # file with all names of classes
batch_size = 50  # how many to process together
epoch_step_count = 2000
epoch_iterations = 10
image_dimesions = (32, 32)
test_ratio = 0.2  # if 1000 images split will 200 for testing
validation_ratio = 0.2  # if 1000 images 20% of remaining 800 will be 160 for validation
###################################################


############################### Importing of the Images
count = 0
images = []
class_ids = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
class_total = len(myList)
print("Importing Classes.....")
for class_id in range(0, len(myList)):
    myImgList = os.listdir(path + "/" + str(count))
    for image_name in myImgList:
        current_image = cv2.imread(path + "/" + str(count) + "/" + image_name, cv2.IMREAD_GRAYSCALE)  # Tek kanallı yapıyoruz..
        current_image = cv2.resize(current_image, (32, 32))  # Boyutlar eşitleniyor.
        images.append(current_image)
        class_ids.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
class_ids = np.array(class_ids)

print(images.shape)
print(images[0])
print(images.dtype)### Veri type ####


########################################## Split Data
X_train, X_test, y_train, y_test = train_test_split(images, class_ids, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validation_ratio)

# X_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID

############################### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("Data Shapes")
print("Train", end="");
print(X_train.shape, y_train.shape)
print("Validation", end="");
print(X_validation.shape, y_validation.shape)
print("Test", end="");
print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[
    0]), "The number of images in not equal to the number of lables in training set"
assert (X_validation.shape[0] == y_validation.shape[
    0]), "The number of images in not equal to the number of lables in validation set"
assert (X_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert (X_train.shape[1:] == (image_dimesions)), " The dimesions of the Training images are wrong "
assert (X_validation.shape[1:] == (image_dimesions)), " The dimesionas of the Validation images are wrong "
assert (X_test.shape[1:] == (image_dimesions)), " The dimesionas of the Test images are wrong"

############################### READ CSV FILE
dataframe = pd.read_csv(label_file, encoding='ISO-8859-1')

print("data shape ", dataframe.shape, type(dataframe))

############################### DISPLAY SOME SAMPLES IMAGES  OF ALL THE CLASSES
sample_counts = []
cols = 5
num_classes = class_total
plt_figure, plt_axes = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
plt_figure.tight_layout()
for i in range(cols):
    for j, row in dataframe.iterrows():
        x_selected = X_train[y_train == j]
        plt_axes[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        plt_axes[j][i].axis("off")
        if i == 2:
            plt_axes[j][i].set_title(str(j) + "-" + row["Name"])
            sample_counts.append(len(x_selected))

############################### DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
print(sample_counts)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), sample_counts)
plt.title("Distribution of the training dataset")
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.show()


############################### PREPROCESSING THE IMAGES




def histogram_equalization(img):
    img = cv2.equalizeHist(img)
    return img


def preprocess_image(img):
    img = histogram_equalization(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img


X_train = np.array(list(map(preprocess_image, X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
X_validation = np.array(list(map(preprocess_image, X_validation)))
X_test = np.array(list(map(preprocess_image, X_test)))
cv2.imshow("GrayScale Images",
           X_train[random.randint(0, len(X_train) - 1)])  # TO CHECK IF THE TRAINING IS DONE PROPERLY

############################### ADD A DEPTH OF 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

############################### AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
image_data_generator = ImageDataGenerator(width_shift_range=0.1,
                                          # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                                          height_shift_range=0.1,
                                          zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                                          shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                                          rotation_range=10)  # DEGREES
image_data_generator.fit(X_train)
data_batches = image_data_generator.flow(X_train, y_train,
                                         batch_size=20)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
X_batch, y_batch = next(data_batches)

# TO SHOW AGMENTED IMAGE SAMPLES
plt_figure, plt_axes = plt.subplots(1, 15, figsize=(20, 5))
plt_figure.tight_layout()

for i in range(15):
    plt_axes[i].imshow(X_batch[i].reshape(image_dimesions[0], image_dimesions[1]))
    plt_axes[i].axis('off')
plt.show()

y_train = to_categorical(y_train, class_total)
y_validation = to_categorical(y_validation, class_total)
y_test = to_categorical(y_test, class_total)


############################### CONVOLUTION NEURAL NETWORK MODEL
def myModel():
    num_filters = 64
    filter_size = (5, 5)  # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
    # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    filter_size2 = (3, 3)
    pool_size = (2, 2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    num_nodes = 500  # NO. OF NODES IN HIDDEN LAYERS
    model = Sequential()
    model.add((Conv2D(num_filters, filter_size, input_shape=(image_dimesions[0], image_dimesions[1], 1),
                      activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(num_filters, filter_size, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

    model.add((Conv2D(num_filters // 2, filter_size2, activation='relu')))
    model.add((Conv2D(num_filters // 2, filter_size2, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_nodes, activation='relu'))
    model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(class_total, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model




############################### TRAIN
model = myModel()
print(model.summary())
history=model.fit(image_data_generator.flow(X_train, y_train, batch_size=batch_size),
                  epochs=20, validation_data=(X_validation,y_validation))


############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])



# Save the model
model.save("saved_model.h5")

# Reload the model
from keras.models import load_model
yeniden_yuklenen_model = load_model("model_kaydedilen.h5")

# Save the model
# model.save("saved_model.keras")

# Reload the model
# from keras.models import load_model
# reloaded_model = load_model("saved_model.keras")