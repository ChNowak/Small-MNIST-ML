'''Data Exploration of the MNIST Dataset
with only 5,6,7,8'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILEPATH=""

mnist_train = pd.read_csv(FILEPATH + "MNIST_train (1).csv").drop("Unnamed: 0", axis = 1)
mnist_test = pd.read_csv(FILEPATH + "MNIST_test (1).csv").drop("Unnamed: 0", axis = 1)

def show_samp_zeros():
    '''display a sample of zeros from the dataset'''

    zeros = mnist_train[mnist_train['response']==0].sample(4)
    zeros = zeros.reset_index().drop("index", axis = 1)
    for i in range(4):
        img_data = zeros.iloc[[i]].drop('response', axis = 1)
        plt.subplots(figsize=(5, 3))
        img_array = np.array(img_data, dtype='uint8').reshape(28, 28)
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')
        plt.title('Number 0')
        plt.show()

def show_samp_fives():
    '''display a sample of fives from the dataset'''

    fives = mnist_train[mnist_train['response']==5].sample(4)
    fives = fives.reset_index().drop("index", axis = 1)
    for i in range(4):
        img_data = fives.iloc[[i]].drop('response', axis = 1)
        plt.subplots(figsize=(5, 3))
        img_array = np.array(img_data, dtype='uint8').reshape(28, 28)
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')
        plt.title('Number 5')
        plt.show()

def show_samp_sixes():
    '''display a sample of sixes from the dataset'''

    sixes = mnist_train[mnist_train['response']==6].sample(4)
    sixes = sixes.reset_index().drop("index", axis = 1)
    for i in range(4):
        img_data = sixes.iloc[[i]].drop('response', axis = 1)
        plt.subplots(figsize=(5, 3))
        img_array = np.array(img_data, dtype='uint8').reshape(28, 28)
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')
        plt.title('Number 6')
        plt.show()

def show_samp_eights():
    '''display a sample of eights from the datset'''

    eights = mnist_train[mnist_train['response']==8].sample(4)
    eights = eights.reset_index().drop("index", axis = 1)
    for i in range(4):
        img_data = eights.iloc[[i]].drop('response', axis = 1)
        plt.subplots(figsize=(5, 3))
        img_array = np.array(img_data, dtype='uint8').reshape(28, 28)
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')
        plt.title('Number 8')
        plt.show()

show_samp_zeros()
show_samp_fives()
show_samp_sixes()
show_samp_eights()