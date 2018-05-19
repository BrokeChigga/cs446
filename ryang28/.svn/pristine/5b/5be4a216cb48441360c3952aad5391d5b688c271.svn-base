"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    data['image'] = None
    data['label'] = None
    pass
    image = []
    label = []
    txt_file = open(data_txt_file,'r')        
    #image_file = open(image_data_path, 'r')
    for line in txt_file:
        content = line.split(',')    
        img_name = content[0]+".jpg"
        img = io.imread(image_data_path+'/' + img_name)
        image.append(img)
        label.append(int(content[1]))

    data['image'] = np.array(image)
    data['label'] = np.reshape(np.array(label), (len(label) ,1))
    return data
