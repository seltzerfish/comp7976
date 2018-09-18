from os import listdir
from helpers import extract_features_ascii_unigram, create_folder, normalize

INPUT_FOLDER = "sports_writers"
OUT_FOLDER = "sports_writers_features"

create_folder(OUT_FOLDER + "-raw")
create_folder(OUT_FOLDER + "-normalized")
files = listdir(INPUT_FOLDER)

for file in files:
    with open(INPUT_FOLDER + "/" + file, "r", encoding='utf-8', errors='ignore') as f:
        # TODO: should this be converted to all lowercase?
        lines = "\n".join(f.readlines())
        features = extract_features_ascii_unigram(lines)
        normalized_features = normalize(features)
        with open(OUT_FOLDER + "-raw/rawcu-" + file, "w") as w:
            w.write(str(features))
        with open(OUT_FOLDER + "-normalized/ncu-" + file, "w") as w:
            w.write(str(normalized_features))
