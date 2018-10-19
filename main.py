from os import listdir
from copy import deepcopy
from pprint import pprint
from helpers import *
from sd_helpers import extract_features
from classifiers import *

INPUT_FOLDER = "sports_writers"
FUNCTIONS = (k_nearest, k_nearest_weighted, grnn, knn_improved, knn_weighted_improved)


# Our collected sports writers:
x_collected, y_collected = load_data_as_x_and_y(INPUT_FOLDER)
x_collected_alt, y_collected_alt = load_data_as_x_and_y(
    INPUT_FOLDER, feature_func=extract_features
)

print("\nSports writers")
print("Number of classes: {}".format(len(set(y_collected))))
for func in FUNCTIONS:
    for k in (1, 3, 5, len(x_collected) - 1):
        if k == len(x_collected) - 1 and func in (k_nearest, knn_improved):
            continue  # dont need to consider k=n for knn
        print(
            '\t"{}" with k = {}: {:.2f}% accuracy.'.format(
                func.__name__,  # function name
                k,  # used k
                score_function_accuracy(
                    func, deepcopy(x_collected), deepcopy(y_collected), k
                )
                * 100,
            )
        )
for k in (1, 3, 5, len(x_collected_alt)):
    print(
        '\t"grnn_improved" with k = {}: {:.2f}% accuracy.'.format(
            k,
            score_function_accuracy(
                grnn, deepcopy(x_collected_alt), deepcopy(y_collected_alt), k
            )
            * 100,
        )
    )
print("\n")


# CASIS 25
x_given, y_given = load_given_features("hw2_data_ncu.txt")
print("CASIS 25")
print("Number of classes: {}".format(len(set(y_given))))
for func in FUNCTIONS:
    for k in (1, 3, 5, len(x_given) - 1):
        if k == len(x_given) - 1 and func in (k_nearest, knn_improved):
            continue  # dont need to consider k=n for knn
        print(
            '\t"{}" with k = {}: {:.2f}% accuracy.'.format(
                func.__name__,
                k,
                score_function_accuracy(func, deepcopy(x_given), deepcopy(y_given), k)
                * 100,
            )
        )
x_given_alt_scoring, y_given_alt_scoring = load_data_as_x_and_y(
    "casis_samples", author_parser=parse_casis_author, feature_func=extract_features
)
for k in (1, 3, 5, len(x_given_alt_scoring)):
    print(
        '\t"grnn_improved" with k = {}: {:.2f}% accuracy.'.format(
            k,
            score_function_accuracy(
                grnn, deepcopy(x_given_alt_scoring), deepcopy(y_given_alt_scoring), k
            )
            * 100,
        )
    )

print()
