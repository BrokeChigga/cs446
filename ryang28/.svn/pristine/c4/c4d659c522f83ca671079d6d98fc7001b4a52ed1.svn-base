"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset_tf(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1], 
                                                     [1, x2], 
                                                     [1, x3],
                                                     .......] 
                                where xi is the 16-dimensional feature of each sample
            
        T(numpy.ndarray): class label vector T = [[y1], 
                                                  [y2], 
                                                  [y3], 
                                                   ...] 
                             where yi is 1/0, the label of each sample 
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    pass
    idx_file = open(path_to_dataset_folder+'/'+index_filename,'r')
    A = []
    T = []
    for line in idx_file:
        content = line.split()
        T.append([int(content[0])])
        sample_file = open(path_to_dataset_folder+'/'+ content[1], 'r')
        row = [1.0]
        files = sample_file.readline().split()
        files = [float(i) for i in files]
        row = row + files
        A.append(np.array(row))
    A = np.array(A)
    T = np.array(T)
    return (A,T)
