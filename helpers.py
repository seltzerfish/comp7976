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

    ascii_chars = list(printable)
    feature_vector = []
    for char in ascii_chars:
        # count occurence of each ascii character
        feature_vector.append(input_string.count(char)) 
    
    # normalize
    mag = magnitude(feature_vector)
    normalized = [x / mag for x in feature_vector]

    return normalized

