import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from feature1 import lemmatize, preprocess

# Feature 2: Number of intersection of lemmas without stop words


def compute_feature2(frases1, frases2, X_train_or_test):
    sw = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    feature = []

    for sent1,sent2 in zip(frases1,frases2):
        sent1 = preprocess(sent1,wnl,sw)
        sent2 = preprocess(sent2,wnl,sw)
        intersect = len(set(sent1).intersection(set(sent2)))
        feature.append(intersect)

    X_train_or_test = np.concatenate((X_train_or_test, np.array(feature).reshape(len(feature),1)), axis=1)
    return X_train_or_test