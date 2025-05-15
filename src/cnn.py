import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from src.subset import subset_dataset
from sklearn.metrics import confusion_matrix, classification_report


def cnn_function(original_data, subset_data):

    # if you want to use original data and not subset, replace subset_data with original_data
    # Replace database path
    database = subset_data
    image_groups = [
        folder for folder in os.listdir(database) if not folder.endswith(".txt")
    ]
    print(image_groups)

    for label in image_groups:
        folder_path = os.path.join(database, label)
        image_files = os.listdir(folder_path)

        if image_files:
            first_img_path = os.path.join(folder_path, image_files[0])
            img = cv2.imread(first_img_path)
            print(img.shape)

            # plt.imshow(img)
            # plt.title(f"{label} - {img.shape}")
            # plt.axis('on')
            # plt.show()

    dataGenerator = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    trainData = dataGenerator.flow_from_directory(
        database,
        target_size=(256, 256),
        batch_size=128,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    valData = dataGenerator.flow_from_directory(
        database,
        target_size=(256, 256),
        batch_size=128,
        class_mode="categorical",
        shuffle=False,
        subset="validation",
    )

    print(trainData)
    print(valData)

    CNN = tf.keras.models.Sequential()
    print(CNN)

    CNN.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, activation="relu", input_shape=[256, 256, 3]
        )
    )
    CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    CNN.add(tf.keras.layers.Flatten())
    CNN.add(tf.keras.layers.Dense(units=512, activation="relu"))
    CNN.add(tf.keras.layers.Dense(units=5, activation="softmax"))
    CNN.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    History = CNN.fit(x=trainData, validation_data=valData, epochs=7)

    # (x_train, y_train), (x_test, y_test) = location
    # x_train
    # pd.DataFrame(History.history)[['accuracy','val_accuracy']].plot()
    # plt.title("Accuracy")
    # plt.show()

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(History.history["accuracy"], color="red", marker="o")
    plt.plot(History.history["val_accuracy"], color="green", marker="h")
    plt.title("Accuracy comparison between Validation and Train Data set", fontsize=15)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="lower right")
    # Change directory to yours
    plt.savefig(
        "Results/20-per-128-batch-7-epoch-accuracy_plot-new.png"
    )  # Save the figure
    plt.show()

    # pd.DataFrame(History.history)[['loss','val_loss']].plot()
    # plt.title("Loss")
    # plt.show()

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(History.history["loss"], color="Purple", marker="o")
    plt.plot(History.history["val_loss"], color="Orange", marker="h")
    plt.title("Loss comparison between Validation and Train Data set", fontsize=15)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="best")
    # Change directory to yours
    plt.savefig("Results/20-per-128-batch-7-epoch-loss_plot-new.png")  # Save the figure
    plt.show()
    # Step 1: Predict on validation data
    valData.reset()  # Ensure proper order of batches
    predictions = CNN.predict(valData, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = valData.classes

    # Step 2: Get class labels
    class_labels = list(valData.class_indices.keys())

    # Step 3: Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("CNN Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(
        "Results/20-per-128-batch-7-epoch-cnn_confusion_matrix.png"
    )  # Save the figure
    plt.show()

    # Step 4: Classification report
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_labels))
