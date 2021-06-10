import gzip
import json
import pickle

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

actions = [1,2,3,4,5]


def save_qtable(qtable):
    print("Saving qtable...")
    f = open("qtable.pkl", "wb")
    pickle.dump(qtable, f)
    f.close()
    print("Done!")


def open_qtable():
    print("Loading qtable...")
    f = open("qtable.pkl", "rb")
    qtable = pickle.load(f)
    print("Done!")
    return qtable


def process_qtable(qtable, states):
    print_qtable = []
    for s in states:
        tmp_row = []
        for a in actions:
            tmp_row.append(qtable[(s, a)])
        print_qtable.append(tmp_row)

    return print_qtable


def load_data():
    print("Loading dataset...")
    json_data = gzip.open("Software.json.gz", 'r')
    n = 2000
    data = []
    for line in json_data:
        tmp = json.loads(line)
        if "reviewText" not in tmp:
            continue
        data.append(tmp)
        if len(data) > n:
            break

    df = pd.DataFrame.from_records(data)[['overall', 'reviewText']]
    print("Done!")
    return df


def process_dataset(t_list):
    print("Processing dataset...")
    count_vector = CountVectorizer(max_features=20000)
    proc_text = count_vector.fit_transform(t_list['reviewText'])

    features = count_vector.get_feature_names()
    # stop_words = count_vector.get_stop_words()
    array = proc_text.toarray()
    print("Done!")
    return array, count_vector


def get_best_action(qtable, state, actions):
    max_val = -999999
    best_action = -1
    val = -1
    for a in actions:
        if (state, a) in qtable:
            val = qtable[(state, a)]
        else:
            qtable[(state, a)] = 0

        if val >= max_val:
            max_val = val
            best_action = a

    if val == -1:
        best_action = np.random.choice(actions)

    assert best_action != -1

    return best_action


def calc_reward(actual, prediction):
    if actual > prediction:
        actual *= -1
    elif prediction > actual:
        prediction *= -1
    else:
        reward = 3
        return reward

    reward = actual + prediction
    return reward


def update(qtable, alpha, reward, state, action):
    qtable[(state, action)] = qtable[(state, action)] + alpha * (reward - qtable[(state, action)])


def train_agent(t_train, s_train):
    # alpha: learning rate
    # epsilon: exploration factor
    # n: number of learning iterations
    print("Starting training process...")
    n = 500
    alpha = 0.1
    epsilon = 0.6
    qtable = {}
    for i in range(n):
        print("Starting run #" + str(i))
        for state, star in zip(t_train, s_train):
            best_action = get_best_action(qtable, state, actions)
            if epsilon > random.uniform(0,1):
                best_action = np.random.choice(actions)
            reward = calc_reward(star, best_action)
            update(qtable, alpha, reward, state, best_action)

        epsilon *= 0.95
    print("Done!")
    return qtable


def test_agent(qtable, states, stars):
    print("Starting testing process...")
    predictions = []
    i = 1
    for state, star in zip(states, stars):
        print("Test #" + str(i) + "/" + str(len(stars)))
        best_state = find_best_match(qtable, state)
        best_val = -99999
        pred_star = -1
        for a in actions:
            val = qtable[(best_state, a)]
            if val > best_val:
                best_val = val
                pred_star = a
        assert pred_star != -1
        predictions.append(pred_star)
        i += 1
    print("Done!")
    return predictions


def find_best_match(qtable, state):
    best_state = 0
    best_val = 999999
    for entry in qtable:
        q_state = entry[0]
        val = 0
        for q, t in zip(q_state, state):
            val += abs(q-t)

        if val < best_val:
            best_val = val
            best_state = q_state

    return best_state


def reinforcement_learning(load=False, test=True):
    print("Starting reinforcement learning!")
    dataframe = load_data()
    freq_array, vectorizer = process_dataset(dataframe)
    stars = dataframe["overall"].values

    text_train, text_test, star_train, star_test = [], [], [], []

    text_train_tmp, text_test_tmp, star_train_tmp, star_test_tmp = train_test_split(freq_array, stars, test_size=0.2, random_state=1)
    for t, s in zip(text_train_tmp, star_train_tmp):
        text_train.append(tuple(t))
        star_train.append(int(s))
    for t, s in zip(text_test_tmp, star_test_tmp):
        text_test.append(tuple(t))
        star_test.append(int(s))

    if not load:
        qtable = train_agent(text_train, star_train)
        save_qtable(qtable)
    else:
        qtable = open_qtable()

    # qtable_print = process_qtable(qtable, text_train)

    # returns tuple (actual star, predicted star)
    if test:
        star_predictions = test_agent(qtable, text_test, star_test)
        print(classification_report(star_test, star_predictions))


reinforcement_learning(load=False, test=True)
