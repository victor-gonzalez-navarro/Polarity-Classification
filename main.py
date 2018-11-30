import numpy as np
import random

from read_training_datasets import read_training_datasets
from read_testing_datasets import read_testing_datasets
from prediction_labels import prediction_labels
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.corpus import treebank

from feature1 import compute_feature1
from feature2 import compute_feature2
from feature3 import compute_feature3
from feature4 import compute_feature4
from feature5 import compute_feature5
from feature6 import compute_feature6
from feature7 import compute_feature7
from feature8 import compute_feature8
from feature9 import compute_feature9
from feature10 import compute_feature10

# ---------------------------------- TRAINING ------------------------------
# Parameters
# best weights = 0.7510791641220215 [4, 15, 3, 0.5, 1, 0.1, 3, 2, 5, 0] rbf 0.3 3 3

possible_weigths = [0, 0.1, 0.2, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 10, 15]
possible_kernels = ['rbf', 'poly', 'linear', 'sigmoid', 'precomputed']
possible_degree = [3, 4, 5, 6, 7]
possible_epsilon = [0.1, 0.1, 0.1, 0.1, 0.2, 0.3]
possible_C = [1, 0.5, 2, 3]
best_result = 0

for a1 in range(5):
    for a2 in range(5):
        for a3 in range(6):
            for a4 in range(4):
                #weights = [possible_weigths[random.randint(0,12)],possible_weigths[random.randint(0,12)],possible_weigths[
                #    random.randint(0,12)],possible_weigths[random.randint(0,12)],possible_weigths[random.randint(0,12)],
                #   #        possible_weigths[random.randint(0,12)],possible_weigths[random.randint(0,12)],possible_weigths[
                #               random.randint(0,12)],possible_weigths[random.randint(0,12)],possible_weigths[random.randint(0,12)]]

                weights = [4, 15, 3, 0.5, 1, 0.1, 3, 2, 5, 0]
                kernel = possible_kernels[a1]
                degree = possible_degree[a2]
                epsilon = possible_epsilon[a3]
                C = possible_C[a4]

                n_components_LDA = 3

                # HMM for feature 9
                trainer = HiddenMarkovModelTrainer()
                st = 3000
                train_data = treebank.tagged_sents()[:st]
                HMM = trainer.train_supervised(train_data)


                # Read training examples and training labels
                N_features = len(weights)
                frases1_train, frases2_train, Y_train = read_training_datasets()
                N_instances_train = len(Y_train)

                # Compute features (X_train)
                X_train = np.zeros((N_instances_train,1))
                X_train = compute_feature1(frases1_train, frases2_train, X_train)
                X_train = compute_feature2(frases1_train, frases2_train, X_train)
                X_train = compute_feature3(frases1_train, frases2_train, X_train)
                X_train = compute_feature4(frases1_train, frases2_train, X_train)
                X_train = compute_feature5(frases1_train, frases2_train, X_train)
                X_train = compute_feature6(frases1_train, frases2_train, X_train)
                X_train = compute_feature7(frases1_train, frases2_train, X_train)
                X_train = compute_feature8(frases1_train, frases2_train, X_train)
                X_train = compute_feature9(frases1_train, frases2_train, X_train, HMM)
                X_train = compute_feature10(frases1_train, frases2_train, X_train)
                X_train = X_train / X_train.max(axis=0)  # Normalize the data
                X_train = X_train * np.array(weights) # Give importance to some features

                # ---------------------------------- TESTING ------------------------------
                # Read testing examples and testing labels
                frases1_test, frases2_test, Y_test = read_testing_datasets()
                N_instances_test = len(Y_test)

                # Compute features (X_test)
                X_test = np.zeros((N_instances_test,1))
                X_test = compute_feature1(frases1_test, frases2_test, X_test)
                X_test = compute_feature2(frases1_test, frases2_test, X_test)
                X_test = compute_feature3(frases1_test, frases2_test, X_test)
                X_test = compute_feature4(frases1_test, frases2_test, X_test)
                X_test = compute_feature5(frases1_test, frases2_test, X_test)
                X_test = compute_feature6(frases1_test, frases2_test, X_test)
                X_test = compute_feature7(frases1_test, frases2_test, X_test)
                X_test = compute_feature8(frases1_test, frases2_test, X_test)
                X_test = compute_feature9(frases1_test, frases2_test, X_test, HMM)
                X_test = compute_feature10(frases1_test, frases2_test, X_test)
                X_test = X_test / X_test.max(axis=0)  # Normalize the data
                X_test = X_test * np.array(weights)  # Give importance to some features

                # Classify
                best_result = prediction_labels(X_train,Y_train,X_test,Y_test,n_components_LDA, best_result, weights, kernel,
                                                degree,epsilon,C)



