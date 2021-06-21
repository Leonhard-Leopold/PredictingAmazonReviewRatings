from method1 import linearsvc
from method2 import supervised_learning
from method3 import reinforcement_learning

method = input("Welcome to our Natural Language Processing project. \nWe are the group 'bezos' and "
               "implemented 3 different methods for predicting Amazon review scores based on the review itself.\n\n"
               "Which method do you want to execute?\n"
               "Enter '1' or 'linearSVC' for the LinearSVC approach\n"
               "Enter '2' or 'Supervised' for the Supervised Learning approach\n"
               "Enter '3' or 'Reinforcement' for the Reinforcement Learning approach"
               "\n:").strip().strip("'").strip('"').lower()
load = input("\nDo you want to load the previously trained model?\n"
             "Enter 'y' or press the enter key to load the previous model.\n"
             "Enter 'n' to train a new model."
             "\n:").strip().strip("'").strip('"').lower()

load = load != "n"
if method == "1" or method == "linearsvc":
    linearsvc(load)
elif method == "2" or method == "supervised":
    supervised_learning(load)
elif method == "3" or method == "reinforcement":
    reinforcement_learning(load=False, test=True, rounds=500, features=200, num_data=8000)
else:
    print("Invalid Input!!")