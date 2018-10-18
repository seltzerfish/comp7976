from os import listdir
from pprint import pprint
from helpers import *
from classifiers import k_nearest, k_nearest_weighted, grnn

INPUT_FOLDER = "sports_writers"
FUNCTIONS = (k_nearest, k_nearest_weighted, grnn)


# Our collected sports writers:
x_collected, y_collected = load_data_as_x_and_y(INPUT_FOLDER)
print("\nSports writers")
print("Number of classes: {}".format(len(set(y_collected))))
for func in FUNCTIONS:
    for k in (1, 3, 5, len(x_collected) - 1):
        print(
            '\t"{}" with k = {}: {:.2f}% accuracy.'.format(
                func.__name__,  # function name
                k,  # used k
                score_function_accuracy(func, x_collected, y_collected, k) * 100,
            )
        )
print("\n")


# CASIS 25
x_given, y_given = load_given_features("hw2_data_ncu.txt")
print("CASIS 25")
print("Number of classes: {}".format(len(set(y_given))))
for func in FUNCTIONS:
    for k in (1, 3, 5, len(x_given) - 1):
        print(
            '\t"{}" with k = {}: {:.2f}% accuracy.'.format(
                func.__name__,
                k,
                score_function_accuracy(func, x_given, y_given, k) * 100,
            )
        )


print()
