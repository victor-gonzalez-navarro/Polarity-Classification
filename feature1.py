import re
import nltk

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Feature 1: Number of intersection of lemmas divided by the maximum length of the two sentences


def compute_feature1(frases1, frases2, X_train_or_test):
    sw = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    feature = []

    for sent1,sent2 in zip(frases1,frases2):
        sent1 = preprocess(sent1,wnl,sw)
        sent2 = preprocess(sent2,wnl,sw)
        intersect = len(set(sent1).intersection(set(sent2)))
        feature.append(intersect/max(len(set(sent1)),len(set(sent2))))

    X_train_or_test[:, 0] = feature
    return X_train_or_test

def lemmatize(p,wnl):
    if p[1][0] in {'N','V'}:
        return wnl.lemmatize(p[0].lower(), pos=p[1][0].lower())
    return p[0]

def preprocess(sent,wnl,sw):
    sent = re.sub(r'[^\w\s]', '', sent)  # remove punctuation
    sent = re.sub("\d+", "", sent)  # remove digits

    sent_tokenized = nltk.word_tokenize(sent)
    sent_postagged = pos_tag(sent_tokenized)
    sent_lemmatized = [lemmatize(pair, wnl) for pair in sent_postagged]
    sent_lemmatized_noSw = [item.lower() for item in sent_lemmatized if (item.lower() not in sw)]
    return sent_lemmatized_noSw