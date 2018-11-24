import re
import nltk
import numpy as np

from nltk.metrics import jaccard_distance
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Feature 5: Number of intersection of words divided by the maximum length of the two sentences


def compute_feature5(frases1, frases2, X_train_or_test):
    sw = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    feature = []

    for sent1,sent2 in zip(frases1,frases2):
        sent1 = preprocess(sent1,wnl,sw)
        sent2 = preprocess(sent2,wnl,sw)
        jaccard_distance(set(sent1), set(sent2))
        feature.append(jaccard_distance(set(sent1), set(sent2)))

    X_train_or_test = np.concatenate((X_train_or_test, np.array(feature).reshape(len(feature), 1)), axis=1)
    return X_train_or_test

def lemmatize(p,wnl):
    if p[1][0] in {'N','V'}:
        return wnl.lemmatize(p[0].lower(), pos=p[1][0].lower())
    return p[0]

def preprocess(sent,wnl,sw):
    sent = re.sub(r'[^\w\s]', '', sent)  # remove punctuation
    sent = re.sub("\d+", "", sent)  # remove digits

    sent_tokenized = nltk.word_tokenize(sent)
    return sent_tokenized