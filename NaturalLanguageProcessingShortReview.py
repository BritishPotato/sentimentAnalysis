# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:18:35 2018

"""
#%% 18 - Improving Training Data for sentiment analysis

"""
So now it is time to train on a new data set. Our goal is to do Twitter 
sentiment, so we're hoping for a data set that is a bit shorter per positive 
and negative statement. It just so happens that I have a data set of 5300+ 
positive and 5300+ negative movie reviews, which are much shorter. 
These should give us a bit more accuracy from the larger training set, as well 
as be more fitting for tweets from Twitter.

I have hosted both files here, you can find them by going to the downloads for 
the short reviews. Save these files as positive.txt and negative.txt.

This process will take a while.. You may want to just go run some errands. 
It took me about 30-40 minutes to run it in full, and I am running an i7 3930k.
For the typical processor in the year I am writing this (2015), it may be hours.
This is a one and done process, however.

"""

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# positive data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

##
### negative data example:      
##training_set = featuresets[100:]
##testing_set =  featuresets[:100]



### Loading classifiers from memory
#classifier = pickle.load(open("NuSVC_classifier", "rb"))
#open("NuSVC_classifier", "rb").close()
#
#
#pickle.dump(classifier, open("naivebayes.pickle", "wb"))
#open("naivebayes.pickle", "wb").close()
#
#pickle.dump(NuSVC_classifier, open("NuSVC_classifier", "wb"))
#open("NuSVC_classifier", "wb").close()
#
#pickle.dump(LinearSVC_classifier, open("LinearSVC_classifier", "wb"))
#open("LinearSVC_classifier", "wb").close()
#    
#pickle.dump(MNB_classifier, open("MNB_classifier", "wb"))
#open("MNB_classifier", "wb").close()
#
#pickle.dump(BernoulliNB_classifier, open("BernoulliNB_classifier", "wb"))
#open("BernoulliNB_classifier", "wb").close()
#    
#pickle.dump(LogisticRegression_classifier, open("LogisticRegression_classifier", "wb"))
#open("LogisticRegression_classifier", "wb").close()



classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


classifier_list = [NuSVC_classifier,
                  LinearSVC_classifier,
                  MNB_classifier,
                  BernoulliNB_classifier,
                  LogisticRegression_classifier]


temp_saved_algo = open("pickled_algos/naivebayes.pickle", "wb")
pickle.dump(classifier, temp_saved_algo)
temp_saved_algo.close()

temp_saved_algo = open("pickled_algos/NuSVC_classifier", "wb")
pickle.dump(NuSVC_classifier, temp_saved_algo)
temp_saved_algo.close()

temp_saved_algo = open("pickled_algos/LinearSVC_classifier", "wb")
pickle.dump(LinearSVC_classifier, temp_saved_algo)
temp_saved_algo.close()
    
temp_saved_algo = open("pickled_algos/MNB_classifier", "wb")
pickle.dump(MNB_classifier, temp_saved_algo)
temp_saved_algo.close()

temp_saved_algo = open("pickled_algos/BernoulliNB_classifier", "wb")
pickle.dump(BernoulliNB_classifier, temp_saved_algo)
temp_saved_algo.close()

temp_saved_algo = open("pickled_algos/LogisticRegression_classifier", "wb")
pickle.dump(LogisticRegression_classifier, temp_saved_algo)
temp_saved_algo.close()



    
    
    
    
    
    
    
    