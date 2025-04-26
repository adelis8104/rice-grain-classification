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
from subset import subset_dataset
'''
from cnn import cnn_function

#I couldnt get the paths to work so change path to your directory
original_data = 'C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Rice_Image_Dataset'
subset_data = 'C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Rice_Subset_20'

cnn_function(original_data, subset_data)
