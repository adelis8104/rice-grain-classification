import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from subset import subset_dataset

original_data = 'C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Rice_Image_Dataset'
subset_data = 'C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Rice_Subset_20'

subset_dataset(original_data, subset_data, percentage=0.2)


# Replace database path
database = subset_data
image_groups = [folder for folder in os.listdir(database) if not folder.endswith('.txt')]
print(image_groups)

for label in image_groups:
    folder_path = os.path.join(database, label)
    image_files = os.listdir(folder_path)

    if image_files:
        first_img_path = os.path.join(folder_path, image_files[0])
        img = cv2.imread(first_img_path)
        print(img.shape)

        plt.imshow(img)
        plt.title(f"{label} - {img.shape}")
        plt.axis('on')
        #plt.show()


dataGenerator = ImageDataGenerator(rescale= 1. / 255, validation_split=0.2)

trainData = dataGenerator.flow_from_directory(
    database,
    target_size=(256,256),
    batch_size=64,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)
valData = dataGenerator.flow_from_directory(
    database,
    target_size=(256,256),
    batch_size=64,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

print(trainData)
print(valData)

CNN = tf.keras.models.Sequential()
print(CNN)

CNN.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',input_shape=[256,256,3]))
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
CNN.add(tf.keras.layers.Flatten())
CNN.add(tf.keras.layers.Dense(units=512, activation='relu'))
CNN.add(tf.keras.layers.Dense(units=5, activation='softmax'))
CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

History = CNN.fit(x=trainData, validation_data=valData, epochs=3)

# (x_train, y_train), (x_test, y_test) = location
# x_train
# pd.DataFrame(History.history)[['accuracy','val_accuracy']].plot()
# plt.title("Accuracy")
# plt.show()

plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
plt.plot(History.history['accuracy'],color="red",marker='o')
plt.plot(History.history['val_accuracy'],color='green',marker='h')
plt.title('Accuracy comparison between Validation and Train Data set',fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig('20-per-accuracy_plot.png')  # Save the figure
plt.show()


# pd.DataFrame(History.history)[['loss','val_loss']].plot()
# plt.title("Loss")
# plt.show()


plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
plt.plot(History.history['loss'],color="Purple",marker='o')
plt.plot(History.history['val_loss'],color='Orange',marker='h')
plt.title('Loss comparison between Validation and Train Data set',fontsize=15)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.savefig('20-per-loss_plot.png')  # Save the figure
plt.show()