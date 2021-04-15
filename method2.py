import gzip
import json
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras import models, layers
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
import re
from keras import backend as BK


def mapping_to_target_range( x, target_min=1, target_max=5) :
    x02 = BK.tanh(x) + 1
    scale = (target_max-target_min)/2.
    return x02 * scale + target_min


def supervised_learning():
    data = []
    g = gzip.open("Software.json.gz", 'r')
    print("Loading dataset ...")
    for l in g:
        data.append(json.loads(l))
    N = 100000
    print("The dataset used has ", len(data), "entries! Of this dataset", N, "entries are used to train the model.")

    reviews = []
    ratings = []
    print("Text preprocessing ...")
    for d in data[:N]:
        if d.get('reviewText') is None:
            continue
        review = re.compile(r'[^a-z0-9\s]').sub(r'', re.compile(r'[\W]').sub(r' ', d.get('reviewText').lower()))
        reviews.append(review)
        rating = float(d.get('overall'))
        ratings.append(rating)

    max_features = 20000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(reviews)
    reviews = tokenizer.texts_to_sequences(reviews)

    max_length = max(len(train_r) for train_r in reviews)
    reviews = tf.keras.preprocessing.sequence.pad_sequences(reviews, maxlen=max_length)

    print("Splitting dataset ...")
    train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(np.array(reviews), np.array(ratings), test_size=0.1)
    train_reviews, validation_reviews, train_ratings, validation_ratings = train_test_split(train_reviews, train_ratings, test_size=0.2)

    input = layers.Input(shape=(max_length,))
    x = layers.Embedding(max_features, 64)(input)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(3)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(1, activation=mapping_to_target_range)(x)
    model = models.Model(inputs=input, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error')

    print("Training Model:\n")
    model.fit(train_reviews, train_ratings, batch_size=128, epochs=2, validation_data=(validation_reviews, validation_ratings))

    preds = model.predict(test_reviews)
    print("\nPredictions:\n")
    print(preds)

    print('Accuracy score: {:0.4}'.format(accuracy_score(test_ratings, 1* (preds>0.5))))

    print("\nTesting model: ")
    x = "I really like this book. It is one of the best I have read."
    x = re.compile(r'[^a-z\s]').sub(r'', re.compile(r'[\W]').sub(r' ', x.lower()))
    x = tokenizer.texts_to_sequences(np.array([x]))
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
    result = model.predict(x)
    print("'I really like this book. It is one of the best I have read.' got the rating: ",  result)

    x = "I really hate this book. It is one of the worst I have read."
    x = re.compile(r'[^a-z\s]').sub(r'', re.compile(r'[\W]').sub(r' ', x.lower()))
    x = tokenizer.texts_to_sequences(np.array([x]))
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
    result = model.predict(x)
    print("'I really hate this book. It is one of the worst I have read.' got the rating: ",  result)

    x = "This book is ok. It is very average."
    x = re.compile(r'[^a-z\s]').sub(r'', re.compile(r'[\W]').sub(r' ', x.lower()))
    x = tokenizer.texts_to_sequences(np.array([x]))
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
    result = model.predict(x)
    print("'This book is ok. It is very average.' got the rating: ",  result)


