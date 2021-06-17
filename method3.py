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


###########################################################
# Saves the learned qtable to a .pkl file
# Parameters:
#   - qtable (dict object to save)
# Returns
#   -
def save_qtable(qtable):
    print("Saving qtable...")
    f = open("qtable.pkl", "wb")
    pickle.dump(qtable, f)
    f.close()
    print("Done!")


###########################################################
# Opens a saved qtable .pkl file
# Parameters:
#   -
# Returns
#   - qtable (loaded dict object)
def open_qtable():
    print("Loading qtable...")
    f = open("qtable.pkl", "rb")
    qtable = pickle.load(f)
    print("Done!")
    return qtable


###########################################################
# Returns the qtable as a printable array
# Parameters:
#   - qtable (dict object to convert)
#   - states (states tuples)
# Returns
#   - print_qtable (array from qtable dict)
def process_qtable(qtable, states):
    print_qtable = []
    for s in states:
        tmp_row = []
        for a in actions:
            tmp_row.append(qtable[(s, a)])
        print_qtable.append(tmp_row)

    return print_qtable


###########################################################
# Loads dataset from .json archive file and saves
# 'num_data' amount of it into an dataframe
# Parameters:
#   - num_data (amount of reviews to load)
# Returns
#   - df (dataframe with reviews and stars)
def load_data(num_data):
    print("Loading dataset...")
    json_data = gzip.open("Software.json.gz", 'r')
    data = []
    star_count = [0, 0, 0, 0, 0]
    for line in json_data:
        tmp = json.loads(line)
        if "reviewText" not in tmp:
            continue

        data.append(tmp)
        if len(data) > num_data:
            break

    df = pd.DataFrame.from_records(data)[['overall', 'reviewText']]
    print("Done!")
    return df


###########################################################
# Converts review text into a word count vector with
# 'features' amount of different words
# Parameters:
#   - t_list (list of reviews)
#   - features (number of different words in count vector)
# Returns
#   - array (word count vector)
#   - count_vector (CountVectorizer object)
def process_dataset(t_list, features):
    print("Processing dataset...")
    count_vector = CountVectorizer(max_features=features)
    proc_text = count_vector.fit_transform(t_list['reviewText'])

    #features = count_vector.get_feature_names()
    array = proc_text.toarray()
    print("Done!")
    return array, count_vector


###########################################################
# Searches for the best star to predict in the qtable,
# based on its entries
# Parameters:
#   - qtable (dict object)
#   - state (state tuple for which the best action is searched)
#   - actions (list of possible actions)
# Returns
#   - best_action (star with highest qtable value)
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


###########################################################
# Returns the reward based on predicted and actual star amount
# Parameters:
#   - actual (actual star amount)
#   - prediction (predicted star amount)
# Returns
#   - reward (integer representing the reward)
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


###########################################################
# Updates the qtable based on RL formula
# Parameters:
#   - qtable (dict object)
#   - alpha (learning rate)
#   - reward (integer representing the reward)
#   - state (state tuple)
#   - action (star amount)
# Returns
#   -
def update(qtable, alpha, reward, state, action):
    qtable[(state, action)] = qtable[(state, action)] + alpha * (reward - qtable[(state, action)])


###########################################################
# Main training function. Iterates over training set and
# first predicts stars randomly. As rounds go on, it chooses
# stars based on best action in qtable. Fills qtable.
# Parameters:
#   - t_train (training word vector list, i.e. states)
#   - s_train (training star list, i.e. actions )
#   - rounds (number of rounds to train)
# Returns
#   - qtable (dict object)
def train_agent(t_train, s_train, rounds):
    # alpha: learning rate
    # epsilon: exploration factor
    print("Starting training process...")
    alpha = 0.1
    epsilon = 0.6
    qtable = {}
    for i in range(rounds):
        #print("Starting run #" + str(i))
        for state, star in zip(t_train, s_train):
            best_action = get_best_action(qtable, state, actions)
            if epsilon > random.uniform(0,1):
                best_action = np.random.choice(actions)
            reward = calc_reward(star, best_action)
            update(qtable, alpha, reward, state, best_action)

        epsilon *= 0.95
    print("Done!")
    return qtable


###########################################################
# Main testing function. Iterates over testing data set and
# chooses best action based on qtable
# Parameters:
#   - qtable (dict object)
#   - states (test states list)
#   - stars (test star list)
# Returns
#   - predictions (list of predicted star values)
def test_agent(qtable, states, stars):
    print("Starting testing process...")
    predictions = []
    i = 1
    correct = 0
    wrong = 0
    for state, star in zip(states, stars):
        #print("Test #" + str(i) + "/" + str(len(stars)))
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
        #print("Pred:" + str(pred_star) + " | " + str(star))
        if pred_star == star:
            correct += 1
        else:
            wrong += 1
        i += 1
    print("Done!")
    print("Right: " + str(correct) + " | Wrong: " + str(wrong))

    return predictions


###########################################################
# Searches qtable for state which matches the input state best
# Parameters:
#   - qtable (dict object)
#   - state (state to match)
# Returns
#   - best_state (state which matches input state the best)
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


###########################################################
# Main reinforcement learning function. Calls all relevant
# functions and prints classification report
# Parameters:
#   - load (Bool if a qtable should be loaded)
#   - test (Bool if RL agent should test)
#   - rounds (number of rounds to train)
#   - features (number of different words in count vector)
#   - num_data (amount of reviews to load)
# Returns
#   -
def reinforcement_learning(load=False, test=True, rounds=500, features=200, num_data=2000):
    print("Starting reinforcement learning!")
    dataframe = load_data(num_data)
    freq_array, vectorizer = process_dataset(dataframe, features)
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
        qtable = train_agent(text_train, star_train, rounds)
        save_qtable(qtable)
    else:
        qtable = open_qtable()

    # qtable_print = process_qtable(qtable, text_train)
    # returns tuple (actual star, predicted star)
    if test:
        star_predictions = test_agent(qtable, text_test, star_test)
        print(classification_report(star_test, star_predictions))
