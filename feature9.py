import numpy as np
import math
import re
import nltk

from feature1 import lemmatize
from nltk.metrics import jaccard_distance
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Feature 9: Synsets


def compute_feature9(frases1, frases2, X_train_or_test, HMM):
    feature = []
    wnl = WordNetLemmatizer()
    sw = set(stopwords.words('english'))

    for sent1,sent2 in zip(frases1,frases2):
        result1 = method_our3(sent1, HMM, sw, wnl)
        result2 = method_our3(sent2, HMM, sw, wnl)
        a = len(set(result1).intersection(set(result2)))
        b = math.sqrt(len(set(result1)) * len(set(result2)))
        feature.append(a / b)

    X_train_or_test = np.concatenate((X_train_or_test, np.array(feature).reshape(len(feature),1)), axis=1)
    return X_train_or_test


def method_our3(dat, HMM, sw,wnl):
    result = []
    k = re.sub(r'[^\w\s]', '', dat)  # remove punctuation
    k2 = re.sub("\d+", "", k)  # remove digits

    sent = nltk.word_tokenize(k)
    sent = pos_tag(sent)
    sent = [lemmatize(pair,wnl) for pair in sent]
    sentmod = [item.lower() for item in sent if (item.lower() not in sw)]
    pairr = HMM.tag(sentmod)
    for pair in pairr:
        var = pair[1][0].lower()
        result.append(lesk_our(sentmod, pair[0], var))
    return (result)


def dista_m(context, ultra):
    ultra = set(ultra)
    # r = len(context.intersection(ultra))
    # r = (len(context.intersection(ultra)))/min(len(context),len(ultra))
    r = jaccard_distance(context, ultra)
    return r


def lesk_our(context_sentence, ambiguous_word, pos=None, synsets=None):
    context = set(context_sentence)
    if synsets is None:
        synsets = wn.synsets(ambiguous_word)
    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]
    if not synsets:
        return None

    numo = 0
    maximo = -1
    for ss in synsets:
        ultra_gloss = []
        # for exampl in ss.examples():
        #    ultra_gloss = ultra_gloss + exampl.split()
        for hyper in ss.hyponyms():
            # for exampl2 in hyper.examples():
            #    ultra_gloss = ultra_gloss + exampl2.split()
            ultra_gloss = ultra_gloss + hyper.definition().split()

        ultra = ss.definition().split() + ultra_gloss
        num_intersection = dista_m(context, ultra)

        if num_intersection > maximo:
            maximo = num_intersection
            idxmax = numo
        numo = numo + 1

    sense2 = synsets[idxmax]

    return sense2

def lemmatize(p,wnl):
    if p[1][0] in {'N','V'}:
        return wnl.lemmatize(p[0].lower(), pos=p[1][0].lower())
    return p[0]