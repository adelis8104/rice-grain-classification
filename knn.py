import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def knn_classify_images(dataset_dir, image_size, k, test_size):
    X = []
    y = []
    class_labels = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for label_idx, label_name in enumerate(class_labels):
        class_path = os.path.join(dataset_dir, label_name)
        for image_name in os.listdir(class_path):
            if image_name.endswith('.jpg'):
                img_path = os.path.join(class_path, image_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                img = img.flatten()  # Flatten into a vector
                X.append(img)
                y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"KNN classification accuracy: {acc*100:.2f}% with k={k}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig('C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Results/confusion_matrix.png')  # Save the figure
    plt.show()

    # Classification Report
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # Plot some correct and incorrect predictions
    plot_knn_predictions(X_test, y_test, y_pred, class_labels, image_size)

def plot_knn_predictions(X_test, y_test, y_pred, class_labels, image_size):
    # Convert flattened vectors back to images
    correct_idx = np.where(y_test == y_pred)[0]
    incorrect_idx = np.where(y_test != y_pred)[0]

    # Plot 5 correct predictions
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(correct_idx[:5]):
        img = X_test[idx].reshape(*image_size, 3)
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Correct: {class_labels[y_pred[idx]]}")
        plt.axis('off')

    # Plot 5 incorrect predictions
    for i, idx in enumerate(incorrect_idx[:5]):
        img = X_test[idx].reshape(*image_size, 3)
        plt.subplot(2, 5, i + 6)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {class_labels[y_test[idx]]}\nPred: {class_labels[y_pred[idx]]}")
        plt.axis('off')

    plt.suptitle('Sample Correct and Incorrect KNN Predictions')
    plt.tight_layout()
    plt.savefig('C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Results/knn_predictions.png')  # Save the figure
    plt.show()