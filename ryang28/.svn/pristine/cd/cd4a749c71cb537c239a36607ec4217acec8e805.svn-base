"""Input and output helpers to load in data.
"""

import pickle
import numpy as np
from numpy import genfromtxt


def read_dataset(input_file_path):
    """Read input file in csv format from file.
    In this csv, each row is an example, stored in the following format.
    label, pixel1, pixel2, pixel3...

    Args:
        input_file_path(str): Path to the csv file.
    Returns:
        (1) label (np.ndarray): Array of dimension (N,) containing the label.
        (2) feature (np.ndarray): Array of dimension (N, ndims) containing the
        images.
    """
    # Imeplemntation here.
    features = []
    labels = []
    #input_file = open(input_file_path, 'r')
    #for line in input_file:
    #    content = line.split(",")
    #    labels.append(content[0])
    #    features.append(content[:-1])
    my_data = genfromtxt(input_file_path, delimiter=',')
    (N,b) = my_data.shape  #simple_test.shape = (200,3)
    for i in range(N):
        labels.append(my_data[i,0])
        features.append(my_data[i,1:])
    labels = np.array(labels)
    features = np.array(features)
    return labels, features
