FILENAME = "logs/steady/0steady.txt"

with open(FILENAME, "r") as f1:
    with open(FILENAME + "_short.txt", "w") as f2:
        count = 0
        for line in f1:
            if count % 400 == 0:
                f2.write(line)
            count += 1