import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Feature 3: Number of intersection of words without stop words


def compute_feature3(frases1, frases2, X_train_or_test):
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

def preprocess(sent,wnl,sw):
    sent = re.sub(r'[^\w\s]', '', sent)  # remove punctuation
    sent = re.sub("\d+", "", sent)  # remove digits

    sent_tokenized = nltk.word_tokenize(sent)
    sent_lemmatized_noSw = [item.lower() for item in sent_tokenized if (item.lower() not in sw)]
    return sent_lemmatized_noSw