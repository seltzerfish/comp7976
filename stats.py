from nltk.tokenize import word_tokenize, sent_tokenize
from os import listdir
from helpers import parse_author_s_writers

INPUT_FOLDER = "sports_writers"
files = listdir(INPUT_FOLDER)

stats_dict = dict()  # initialize dictionary

for file in files:
    with open(INPUT_FOLDER + "/" + file, "r", encoding='utf-8', errors='ignore') as f:
        lines = "\n".join(f.readlines())
        author = parse_author_s_writers(file)
        chars = len(lines)
        words = len(word_tokenize(lines))
        sentences = len(sent_tokenize(lines))
        if author not in stats_dict:
            stats_dict[author] = [[chars, words, sentences]]
        else:
            stats_dict[author].append([chars, words, sentences])

for author in stats_dict:
    print(author)
    print("\tAverage # characters: " +
          str(sum([x[0] for x in stats_dict[author]]) / len(stats_dict[author])))
    print("\tAverage # words: " +
          str(sum([x[1] for x in stats_dict[author]]) / len(stats_dict[author])))
    print("\tAverage # sentences: " +
          str(sum([x[2] for x in stats_dict[author]]) / len(stats_dict[author])))
    print()
