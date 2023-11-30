# This is the script to transform the json data to csv

import json
import pandas as pd

train_input = "./train_data.json"
dev_input = "./validation_data.json"
test_input = "./test_data.json"

train_item_list = json.load(open(train_input, 'r'))
dev_item_list = json.load(open(dev_input, 'r'))
test_item_list = json.load(open(test_input, 'r'))

user_history_dict = dict()
train_data = []
item_corpus = []
corpus_index = dict()
for user_id, items in enumerate(train_item_list):
    items = [str(x) for x in items]
    user_history_dict[user_id] = items
    for item in items:
        if item not in corpus_index:
            corpus_index[item] = int(item)
            item_corpus.append([corpus_index[item], item])
        history = user_history_dict[user_id].copy()
        history.remove(item)
        train_data.append([user_id, corpus_index[item], 1, user_id, "^".join(history)])
train = pd.DataFrame(train_data, columns=["query_index", "corpus_index", "label", "user_id", "user_history"])
print("train samples:", len(train))
train.to_csv("train.csv", index=False)

dev_data = []
for user_id, items in enumerate(dev_item_list):
    items = [str(x) for x in items]
    for item in items:
        if item not in corpus_index:
            corpus_index[item] = int(item)
            item_corpus.append([corpus_index[item], item])
        history = user_history_dict[user_id].copy()
        dev_data.append([user_id, corpus_index[item], 1, user_id, "^".join(history)])
dev = pd.DataFrame(dev_data, columns=["query_index", "corpus_index", "label", "user_id", "user_history"])
print("validation samples:", len(dev))
dev.to_csv("valid.csv", index=False)

test_data = []
for user_id, items in enumerate(test_item_list):
    items = [str(x) for x in items]
    for item in items:
        if item not in corpus_index:
            corpus_index[item] = int(item)
            item_corpus.append([corpus_index[item], item])
        history = user_history_dict[user_id].copy()
        test_data.append([user_id, corpus_index[item], 1, user_id, "^".join(history)])
test = pd.DataFrame(test_data, columns=["query_index", "corpus_index", "label", "user_id", "user_history"])
print("test samples:", len(test))
test.to_csv("test.csv", index=False)

corpus = pd.DataFrame(item_corpus, columns=["corpus_index", "item_id"])
print("number of items:", len(item_corpus))
corpus = corpus.set_index("corpus_index").sort_index()
corpus.to_csv("item_corpus.csv", index=False)
