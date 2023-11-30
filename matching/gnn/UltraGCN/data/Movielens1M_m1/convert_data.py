import json

train = json.load(open("train_data.json", "r"))
with open("train.txt", "w") as fout:
    for i, items in enumerate(train):
        fout.write(str(i) + " " + " ".join(map(str, items)) + "\n")

valid = json.load(open("validation_data.json", "r"))
with open("valid.txt", "w") as fout:
    for i, items in enumerate(valid):
        fout.write(str(i) + " " + " ".join(map(str, items)) + "\n")

test = json.load(open("test_data.json", "r"))
with open("test.txt", "w") as fout:
    for i, items in enumerate(test):
        fout.write(str(i) + " " + " ".join(map(str, items)) + "\n")
