import gzip
import json
import pandas as pd


def supervised_learning():
    data = []
    g = gzip.open("Software.json.gz", 'r')
    print("Loading dataset ...")
    for l in g:
        data.append(json.loads(l))
    N = 25000
    data = data[:N]
    print("The dataset used has ", len(data), "entries! Of this dataset", N, "entries are used to train the model.")

    reviews = []
    ratings = []
    for d in data:
        reviews.append(d.get('reviewText'))
        ratings.append(d.get('overall'))

    print("TODO")
