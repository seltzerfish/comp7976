from os import listdir
from helpers import extract_features_ascii_unigram

files = listdir("data")
for file in files:
    with open("data/" + file, "r") as f:
        lines = "\n".join(f.readlines())
        features = extract_features_ascii_unigram(lines)
        with open("features_unigram/features-" + file, "w") as w:
            w.write(str(features))
