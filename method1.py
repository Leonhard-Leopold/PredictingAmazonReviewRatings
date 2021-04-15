import gzip
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def linearsvc():
    data = []
    g = gzip.open("Software.json.gz", 'r')
    print("Loading dataset ...")
    for l in g:
        data.append(json.loads(l))
    N = 100000
    data = data[:N]
    print("The dataset used has ", len(data), "entries! Of this dataset", N, "entries are used to train the model.")

    df = pd.DataFrame.from_records(data)[['overall', 'reviewText']]
    df.fillna("", inplace=True)
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer='char')
    print("Preprocessing ...")
    x = tfidf.fit_transform(df['reviewText'])
    y = df['overall']
    print("splitting dataset ...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    print("training model ...")
    clf = LinearSVC(C = 20, class_weight="balanced", verbose=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    print("Testing model:\n ")
    x = "I really like this book. It is one of the best I have read."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", clf.predict(vec)[0])
    x = "I really hate this book. It is one of the worst I have read."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", clf.predict(vec)[0])
    x = "This book is ok. It is very average."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", clf.predict(vec)[0])

