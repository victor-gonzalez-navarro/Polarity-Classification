import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

# Feature 4: Difference between the positivity of both sentences


def compute_feature4(frases1, frases2, X_train_or_test):
    sw = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    feature = []

    for sent1, sent2 in zip(frases1, frases2):
        sent1 = preprocess(sent1, wnl, sw)
        sent2 = preprocess(sent2, wnl, sw)
        p_score = objectivity(sent1)-objectivity(sent2)
        feature.append(p_score)

    X_train_or_test = np.concatenate((X_train_or_test, np.array(feature).reshape(len(feature), 1)), axis=1)
    return X_train_or_test


def preprocess(sent, wnl, sw):
    sent = re.sub(r'[^\w\s]', '', sent)  # remove punctuation
    sent = re.sub("\d+", "", sent)  # remove digits

    sent_tokenized = nltk.word_tokenize(sent)
    return sent_tokenized

def objectivity(text):
    positivity_score = 0 #score to determine if the file is positive or negative
    pairs = pos_tag(text) #list of tuples of the words from a text file
    pairs = dict(pairs) #dictionary of pairs
    for item in text:
        # Adjectives in POS tag are JJ and in synsets are A:
        if pairs[item][0].lower()== 'j':
            synset = wn.synsets(item,'a')
            # In case the synset does exist
            if synset != []:
                synset = synset[0]
                sentiSynset = swn.senti_synset(synset.name())
                positivity_score = positivity_score + sentiSynset.pos_score() - sentiSynset.neg_score()

        # Adverbs in POS tag are RB and in synsets are R:
        elif pairs[item][0].lower()== 'r':
            synset = wn.synsets(item,'r')
            # In case the synset does exist
            if synset != []:
                synset = synset[0]
                sentiSynset = swn.senti_synset(synset.name())
                positivity_score = positivity_score + sentiSynset.pos_score() - sentiSynset.neg_score()
    return positivity_score

