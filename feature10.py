import re
import nltk
import numpy as np
from nltk.metrics import jaccard_distance
from nltk.parse.corenlp import CoreNLPDependencyParser



# Feature 10: Jaccard distance of dependency triples from CoreNLPDependencyParser


def compute_feature10(frases1, frases2, X_train_or_test):
    feature = []

    for sent1,sent2 in zip(frases1, frases2):
        result1 = method(sent1)
        result2 = method(sent2)
        c = 1 - jaccard_distance(set(result1), set(result2))
        feature.append(c)

    X_train_or_test = np.concatenate((X_train_or_test, np.array(feature).reshape(len(feature),1)), axis=1)
    return X_train_or_test


def method(dat):
    result = []
    k = re.sub(r'[^\w\s]', '', dat)  # remove punctuation
    k2 = re.sub("\d+", "", k)  # remove digits
    parser = CoreNLPDependencyParser(url='http://localhost:9000')

    # Triples from CoreNLPDependencyParser
    sent = k.lower()
    if sent == 'tunisia':
        return 'tunisia'
    else:
        parse = parser.raw_parse(sent)
        tree = next(parse)
        sentparse = [t for t in tree.triples()]
        return sentparse

