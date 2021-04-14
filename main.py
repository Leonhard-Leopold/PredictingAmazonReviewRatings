from method1 import linearsvc
from method2 import supervised_learning
from method3 import reinforcement_learning

method = input("Welcome to our Natural Language Processing project. \nWe are the group 'bezos' and "
               "implemented 3 different methods for predicting Amazon review scores based on the review itself?\n\n"
               "Which method do you want to execute?\n"
               "Enter '1' or 'linearSVC' for the LinearSVC approach\n"
               "Enter '2' or 'Supervised' for the Supervised Learning approach\n"
               "Enter '3' or 'Reinforcement' for the Reinforcement Learning approach"
               "\n:").strip().strip("'").strip('"').lower()

if method == "1" or method == "linearsvc":
    linearsvc()
elif method == "2" or method == "supervised":
    supervised_learning()
elif method == "3" or method == "reinforcement":
    reinforcement_learning()
else:
    print("Invalid Input!!")