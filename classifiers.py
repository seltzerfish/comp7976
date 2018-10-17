from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from copy import deepcopy
from math import exp
from scipy.spatial.distance import euclidean


DEFAULT_SIGMA = 0.11


def k_nearest(target, x, y, k=3):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x, y)
    return classifier.predict(target)


def k_nearest_weighted(target, x, y, k=3):
    classifier = KNeighborsClassifier(n_neighbors=k, weights="distance")
    classifier.fit(x, y)
    return classifier.predict(target)


def grnn(target, x, y, k=3, sigma=DEFAULT_SIGMA):
    population = limit_population(
        deepcopy(x), k, target
    )  # only consider closest k instances
    population_labels = [y[x.index(e)] for e in population]
    classes = list(set(population_labels))  # remove duplicates
    hf_values = calculate_hf_values(target, population, sigma)
    d_values = np.array(calculate_d_values(population_labels, classes))
    numerator = np.array([0.0] * len(classes))
    denominator = sum(hf_values)
    for i in range(len(hf_values)):
        numerator += d_values[i] * hf_values[i]
    prediction_vector = list(numerator / denominator)
    return [classes[prediction_vector.index(max(prediction_vector))]]


## helper methods


def limit_population(x, k, target):
    x.sort(key=lambda e: euclidean(e, target))
    return x[:k]


def calculate_hf_values(target, x, sigma):
    ret = []
    for x_i in x:
        neg_dist = euclidean(target, x_i) * (-1)
        sig_sqrd_2 = (sigma ** 2) * 2  # need to optimize?
        ret.append(exp(neg_dist / sig_sqrd_2))
    return ret


def calculate_d_values(y, classes):
    ret = []
    for label in y:
        vector = [0] * len(classes)
        vector[classes.index(label)] = 1
        ret.append(vector)
    return ret
