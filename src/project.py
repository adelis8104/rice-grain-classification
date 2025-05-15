import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
"""
from src.subset import subset_dataset
from src.cnn import cnn_function
from src.knn3 import knn_classify_images
from src.knn2 import knn_evaluate_multiple_k
from src.hog import hog_function
from src.sift import sift_function

original_data = "Rice_Image_Dataset"
subset_data = "Rice_Subset_20"

# Create the subset dataset That will populate the subset folder
subset_dataset(original_data, subset_data, percentage=0.2)

cnn_function(original_data, subset_data)

knn_classify_images(original_data, image_size=(64, 64), k=7, test_size=0.2)

k_values = list(range(1, 16))  # You can adjust this range as needed
knn_evaluate_multiple_k(
    original_data, image_size=(64, 64), k_values=k_values, test_size=0.2
)

results = hog_function(original_data, test_size=0.2)

results = sift_function(original_data, test_size=0.2)
