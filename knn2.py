from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np

def knn_evaluate_multiple_k(dataset_dir, image_size, k_values, test_size=0.2):
    X = []
    y = []
    class_labels = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    # Load and preprocess images
    for label_idx, label_name in enumerate(class_labels):
        class_path = os.path.join(dataset_dir, label_name)
        for image_name in os.listdir(class_path):
            if image_name.endswith('.jpg'):
                img_path = os.path.join(class_path, image_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                img = img.flatten()
                X.append(img)
                y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    # Evaluate multiple k values
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"k={k} -> Accuracy: {acc*100:.2f}%")
        accuracies.append(acc)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(k_values, accuracies, marker='o', color='blue')
    plt.title("KNN Accuracy vs k")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig('C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Results/accuracy-minkowski.png')
    plt.show()