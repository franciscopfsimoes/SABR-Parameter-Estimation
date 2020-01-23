#Useful Python packages
import numpy as np


def preprocess(file):

    myfile = open(file, "r")

    dataset = myfile.readlines()

    del dataset[0]

    clean_lines = [l.strip('\n') for l in dataset]

    A = [row.split('\t') for row in clean_lines]

    A = np.delete(A, 0, 1)

    dataset = np.array(A)

    return dataset