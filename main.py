from os import listdir
from pprint import pprint
from helpers import *
from classifiers import k_nearest

INPUT_FOLDER = "sports_writers"

x, y = load_data_as_x_and_y(INPUT_FOLDER)

for index in range(len(x)):
    print("actual: " + y[index])
    print("predicted: " + str(k_nearest([x[index]], x, y)))
