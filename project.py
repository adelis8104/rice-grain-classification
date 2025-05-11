import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
'''
from subset import subset_dataset
from cnn import cnn_function
from knn import knn_classify_images
from knn2 import knn_evaluate_multiple_k

#I couldnt get the paths to work so change path to your directory
original_data = 'C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Rice_Image_Dataset'
subset_data = 'C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Rice_Subset_20'

#Create the subset dataset That will populate the subset folder
subset_dataset(original_data, subset_data, percentage=0.2)

cnn_function(original_data, subset_data)

#knn_classify_images(original_data, image_size=(64, 64), k=7, test_size=0.2)

#k_values = list(range(1, 16))  # You can adjust this range as needed
#knn_evaluate_multiple_k(subset_data, image_size=(64, 64), k_values=k_values, test_size=0.2)
