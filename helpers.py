import os
from string import printable
from math import sqrt

def magnitude(vector):
    '''Computes magnitude of vector

    Args:
        vector (iterable): an iterable object of numeric contents

    Returns:
        float: magnitude
    '''

    return sqrt(sum([x ** 2 for x in vector]))


def extract_features_ascii_unigram(input_string):
    '''Extracts feature scores based on single letter occurences of ascii characters.

    Args:
        input_string (string): the string to be scored

    Returns:
        list: a list of feature scores
    '''

    # ascii_chars = list(printable)[:-5] #TODO: should whitespace and newlines be omitted?
    ascii_chars = list(printable)
    feature_vector = []
    for char in ascii_chars:
        # count occurence of each ascii character
        feature_vector.append(input_string.count(char))

    return feature_vector


def normalize(vector):
    '''normalizes a vector
    
    Args:
        vector (iterable): the vector to be normalized
    
    Returns:
        list: normalized vector
    '''


    mag = magnitude(vector)
    return [x / mag for x in vector]


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)