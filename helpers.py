import os
from string import printable
from math import sqrt


def magnitude(vector):
    """Computes magnitude of vector

    Args:
        vector (iterable): an iterable object of numeric contents

    Returns:
        float: magnitude
    """

    return sqrt(sum([x ** 2 for x in vector]))


def extract_features_ascii_unigram(input_string):
    """Extracts feature scores based on single letter occurences of ascii characters.

    Args:
        input_string (string): the string to be scored

    Returns:
        list: a list of feature scores
    """

    # ascii_chars = list(printable)[:-5] #TODO: should whitespace and newlines be omitted?
    ascii_chars = list(printable)
    feature_vector = []
    for char in ascii_chars:
        # count occurence of each ascii character
        feature_vector.append(input_string.count(char))

    return feature_vector


def normalize(vector):
    """normalizes a vector

    Args:
        vector (iterable): the vector to be normalized

    Returns:
        list: normalized vector
    """

    mag = magnitude(vector)
    return [x / mag for x in vector]


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def parse_author_s_writers(filename):
    first_occurence = filename.index("_")
    second_underscore_index = filename.index("_", first_occurence + 1)
    return filename[:second_underscore_index]


def load_data_as_normalized_dict(folder_name):
    """reads in a folder of text, and scores it using character unigram
    
    Args:
        folder_name (str): where the files live
    
    Returns:
        dict: maps author names to a list of sample feature vectors
    """

    data = dict()
    file_names = os.listdir(folder_name)
    for file in file_names:
        with open(
            folder_name + "/" + file, "r", encoding="utf-8", errors="ignore"
        ) as f:
            # TODO: should this be converted to all lowercase?
            lines = "\n".join(f.readlines())
            features = extract_features_ascii_unigram(lines)
            normalized_features = normalize(features)
            author = parse_author_s_writers(file)
            if author in data:
                data[author].append(normalized_features)
            else:
                data[author] = [normalized_features]
    return data


def load_data_as_x_and_y(folder_name, feature_func=extract_features_ascii_unigram):
    """extracts the feature vectors for all files in a folder
    
    Args:
        folder_name (str): where the files live
        feature_func (function, optional): Defaults to extract_features_ascii_unigram. The function to be used to convert a string to a feature vector
    
    Returns:
        list, list: two lists representing each feature vector, and each label associated
    """

    x = []
    y = []
    file_names = os.listdir(folder_name)
    for file in file_names:
        with open(
            folder_name + "/" + file, "r", encoding="utf-8", errors="ignore"
        ) as f:
            # TODO: should this be converted to all lowercase?
            lines = "\n".join(f.readlines())
            features = feature_extraction_func(lines)
            normalized_features = normalize(features)
            author = parse_author_s_writers(file)
            x.append(normalized_features)
            y.append(author)
    return x, y


def load_given_features(file_name):
    x = []
    y = []
    with open(file_name, "r") as f:
        for line in f:
            l = list(eval(line))
            y.append(l[0])
            x.append(l[1:])
    return x, y


def score_function_accuracy(function, x, y, k):
    correct = 0.0
    for index in range(len(x)):
        sample = x[index]
        true_label = y[index]
        del x[index]
        del y[index]
        predicted = function([sample], x, y, k)[0]
        if true_label == predicted:
            correct += 1
        x.insert(index, sample)
        y.insert(index, true_label)
    return correct / (len(x))
