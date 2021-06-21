import gzip
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import joblib
import re


###########################################################
# Main LinearSVC method. Loads the dataset, applies
# preprocessing, trains the model, saves the model
# parameters and prints the  classification report
# Parameters:
#   - load (Bool if the weights should be loaded)
# Returns
#   -
def linearsvc(load):
    # Loading dataset
    data = []
    g = gzip.open("Software.json.gz", 'r')
    print("Loading dataset ...")
    for l in g:
        data.append(json.loads(l))
    N = 100000
    print("The dataset used has ", len(data), "entries! Of this dataset", N, "entries are used to train the model.")
    data = data[:N]

    # creating pandas dataframe with two needed columns
    df = pd.DataFrame.from_records(data)[['overall', 'reviewText']]
    df.fillna("", inplace=True)

    # remove all unwanted chars
    df['reviewText'] = df['reviewText'].map(lambda a: re.compile(r'[^a-z0-9\s]')
                                            .sub(r'', re.compile(r'[\W]').sub(r' ', a.lower())))

    # vectorized the input texts
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer='char')
    print("Preprocessing ...")
    x = tfidf.fit_transform(df['reviewText'])
    y = df['overall']

    # splits the training set into training and test set
    print("splitting dataset ...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # creating the model
    print("training model ...")
    clf = LinearSVC(C = 20, class_weight="balanced", verbose=1)

    # loads the model if wanted
    if load:
        print("\nLoading previous model weights:\n")
        clf = joblib.load("weights/linearSVC.sav")

    # trains the model
    clf.fit(X_train, y_train)

    # predicting the the ratings on the test set
    y_pred = clf.predict(X_test)

    # printing the classification report
    print(classification_report(y_test, y_pred))

    # testing the model with 3 examples (positive, negative, neutral review)
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

    # saving the model weights
    print("\n\nSaving model weights ...")
    joblib.dump(clf, "weights/linearSVC.sav")

