import gzip
import json
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras import models, layers
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import re
from keras import backend as BK


###########################################################
# taking the input, applying the tanh function and scaling
# to output from -1 to 1 to the wanted scale
# Parameters:
#   - x (input float)
#   - target_min (lower bound of the wanted scale)
#   - target_max (upper bound of the wanted scale)
# Returns
#   - scaled input
def mapping_to_target_range(x, target_min=1, target_max=5):
    x02 = BK.tanh(x) + 1
    scale = (target_max - target_min) / 2.
    return x02 * scale + target_min


###########################################################
# Main supervised learning with neural network method.
# Loads the dataset, applies preprocessing, trains the
# model, saves the model parameters and prints the
# classification report
# Parameters:
#   - load (Bool if the weights should be loaded)
# Returns
#   -
def supervised_learning(load):
    # Loading dataset
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
        # remove all unwanted chars
        review = re.compile(r'[^a-z0-9\s]').sub(r'', re.compile(r'[\W]').sub(r' ', d.get('reviewText').lower()))
        reviews.append(review)
        rating = float(d.get('overall'))
        ratings.append(rating)

    # vectorized the input texts
    max_features = 200000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(reviews)
    reviews = tokenizer.texts_to_sequences(reviews)

    # calculating the maximal review length & pad all inputs to the same length for the neural network
    max_length = max(len(train_r) for train_r in reviews)
    reviews = tf.keras.preprocessing.sequence.pad_sequences(reviews, maxlen=max_length)

    # split the data into training set, test set and validation set
    print("Splitting dataset ...")
    train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(np.array(reviews), np.array(ratings),
                                                                                test_size=0.1)
    train_reviews, validation_reviews, train_ratings, validation_ratings = train_test_split(train_reviews,
                                                                                            train_ratings,
                                                                                            test_size=0.2)

    # Create the neural network. Input size was calculated above
    input = layers.Input(shape=(max_length,))
    x = layers.Embedding(max_features, 64)(input)
    # three times, use a convolutional layer, normalization and max pooling layer
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(3)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Flatten()(x)
    # two normally connected layers to condense the output to a single number
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation=mapping_to_target_range)(x)
    model = models.Model(inputs=input, outputs=output)

    # Adam (a stochastic gradient descent variant) as optimization function
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    # compiling the model. As error the MSE is specified since the output and target are floats
    model.compile(optimizer=opt, loss='mean_squared_error')

    # loading model weights if wanted
    if load:
        print("\nLoading previous model weights:\n")
        model.load_weights('weights/supervisedLearning')

    # training the model
    print("Training Model:\n")
    model.fit(train_reviews, train_ratings, batch_size=64, epochs=3,
              validation_data=(validation_reviews, validation_ratings))

    # calculating the predictions on the test set
    test_pred = model.predict(test_reviews)

    # printing the classification report
    print(classification_report(test_ratings,np.round(test_pred)))

    # testing the model with 3 examples (positive, negative, neutral review)
    print("\nTesting model: ")
    x = "I really like this book. It is one of the best I have read."
    x = re.compile(r'[^a-z\s]').sub(r'', re.compile(r'[\W]').sub(r' ', x.lower()))
    x = tokenizer.texts_to_sequences(np.array([x]))
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
    result = model.predict(x)
    print("'I really like this book. It is one of the best I have read.' got the rating: ", round(result[0][0]))

    x = "I really hate this book. It is one of the worst I have read."
    x = re.compile(r'[^a-z\s]').sub(r'', re.compile(r'[\W]').sub(r' ', x.lower()))
    x = tokenizer.texts_to_sequences(np.array([x]))
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
    result = model.predict(x)
    print("'I really hate this book. It is one of the worst I have read.' got the rating: ", round(result[0][0]))

    x = "This book is ok. It is very average."
    x = re.compile(r'[^a-z\s]').sub(r'', re.compile(r'[\W]').sub(r' ', x.lower()))
    x = tokenizer.texts_to_sequences(np.array([x]))
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
    result = model.predict(x)
    print("'This book is ok. It is very average.' got the rating: ", round(result[0][0]))

    # saving the model weights
    print("\n\nSaving model weights ...")
    model.save_weights('weights/supervisedLearning')
