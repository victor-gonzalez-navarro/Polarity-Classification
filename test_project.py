import numpy as np
import pickle

from scipy.stats import pearsonr
from read_testing_datasets import read_testing_datasets
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.corpus import treebank
from joblib import load

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

print('Testing the project with the saved model: ')
weights = [5, 10, 0.1, 0.2, 0.5, 0, 5, 10, 2, 1]

# HMM for feature 9
trainer = HiddenMarkovModelTrainer()
st = 3000
train_data = treebank.tagged_sents()[:st]
HMM = trainer.train_supervised(train_data)


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
filename = ''
loaded_model = load(filename)
pred_4 = loaded_model.predict(X_test)
print('\nPearson Correlation using SVR:')
result = pearsonr(pred_4, Y_test)[0]
print(result)
