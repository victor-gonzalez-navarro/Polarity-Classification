import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.metrics import jaccard_distance


# Feature 7: Jaccard distance of bigrams without stop wprds


def compute_feature7(frases1, frases2, X_train_or_test):
    feature = []
    sw = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    feature = []

    for sent1, sent2 in zip(frases1, frases2):
        sent1b = sent1
        sent2b = sent2
        sent1 = preprocess(sent1, wnl, sw)
        sent2 = preprocess(sent2, wnl, sw)

        bigrams1 = list(nltk.bigrams(sent1))
        bigrams2 = list(nltk.bigrams(sent2))
        if len(bigrams1)==0 or len(bigrams2)==0:
            feature.append(0)
        else:
            feature.append(jaccard_distance(set(bigrams1), set(bigrams2)))

    X_train_or_test = np.concatenate((X_train_or_test, np.array(feature).reshape(len(feature), 1)), axis=1)
    return X_train_or_test


def preprocess(sent,wnl,sw):
    sent = re.sub(r'[^\w\s]', '', sent)  # remove punctuation
    sent = re.sub("\d+", "", sent)  # remove digits

    sent_tokenized = nltk.word_tokenize(sent)
    sent_tokenized = [item.lower() for item in sent_tokenized if (item.lower() not in sw)]
    return sent_tokenized