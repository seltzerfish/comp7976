from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def k_nearest(target, x, y, k=3):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x, y)
    return classifier.predict(target)

def k_nearest_weighted(target, x, y, k=3):
    classifier = KNeighborsClassifier(n_neighbors=k, weights="distance")
    classifier.fit(x, y)
    return classifier.predict(target)