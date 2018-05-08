# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:34:51 2018

@author: British Potato
"""
#%% 1 - Tokenizing Words and Sentences
import nltk

## nltk.download()
## tokenizing (grouping)
## word tokenizer - seperates by word - word.tokenize()
## sentence tokenizer - seperates by sentence - sent.tokenize()

## corpora - body of text
## lexicon - words and their meaning

## e.g. investor-speak "bull" VS regular english-speak "bull"
## investor speak "bull" - someone who is positive about the market
## english-speak "bull" - animal

from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello there Mr. Smith, how do you do? What is the meaning of life?"

#print(sent_tokenize(example_text))
#print(word_tokenize(example_text))


for i in word_tokenize(example_text):
    #print(i)
    continue


## nltk mostly helps with pre-processing i.e. organize the data, such
## as pulling apart the data, tagging/labelling, stop words...

#%% 2 - Stop Words: useless words in analysis, meaningless.
## "the, a, an" are examples
    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is an example showing off stop word filtration."
words = word_tokenize(example_sentence)

stop_words = set(stopwords.words("english"))
## print(stop_words)

filtered_sentence = []

#for w in words:
#    if w not in stop_words:
#        filtered_sentence.append(w)

#filtered_sentence = [w for w in words if w not in stop_words]

print(filtered_sentence)

#%% 3 - Stemming


## Analysis is the thing you do at the very end, the cherry on top.
## Most part of data analysis is the organizing and cleaning of data.

## Stemming is the "normalization" of words e.g. riding, ridden, rode --> ride

## I was taking a ride in the car. = I was riding in the car.
## Stemming ensures that these two are indeed the same.
## Stemming has been around since 1979!

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]
## Let us stem these words

for w in example_words:
    #print(ps.stem(w))
    continue

new_text = "It is very important to pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)

for w in words:
    #print(ps.stem(w))
    continue

## Stemming depends on your goal. You won't actually have to stem, instead
## you will be using wordnet instead. Though you still should know.

#%% 4 - Part of Speech Tagging

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer # Unsupervized ML Sentence Tokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content4():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

#process_content4()

## May not recognize nouns. Causes problems when reading Twitter.
## Lots of people do not capitalize, so names can be lowercase, causing nltk to go "wut"
## We are beginning to derive meaning, but there is still more work to do.

"""
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
"""

#%% 5 - Chunking

## The next step to figuring out the meaning of the sentence is to understand
## what/who is the sentence talking about? The subject. Generally a person, place or thing.

## Then the next step is finding the words with modifier effect. 

## Most people chunk in "noun phrases" i.e. phrases of one or more words that
## contain a noun, maybe some descriptive words, maybe a verb, maybe an adverb.
## The idea is to group nouns with the words which actually relate to them.

## To chunk, we combine the part of speech tags with regular expressions.

def process_content5():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk:{<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            print(chunked)
            
            ## The "chunked" variable is an NLTK tree. Each "chunk" and "non-chunk" is a 
            ## subtree of the tree. We can reference these through chunked.subtrees.
                    
            ## If we want to access this data, iteration through these subtrees can be done as:
            for subtree in chunked.subtrees():
                print(subtree)
                
            ## If we are only interested in just the chunks, we can use filter parameter
            #for subtree in chunked.subtrees(filter=lambda t: t.label() == "Chunk"):
            #    print(subtree)
            
            chunked.draw()
            
    except Exception as e:
        print(str(e))
        
#process_content5()
        
## chunkGram = r"""Chunk:{<RB.?>*<VB.?>*<NNP>+<NN>?}"""
## <RB.?>* = "0 or more of any tense of adverb"
## <VB.?>* = "0 or more of any tense of verb"
## <NNP>+ = "1 or more proper nouns"
## <NN>? = "0 or 1 singular noun"

#%% 6 - Chinking: The reverse of chunking. Removes a chunk from a chunk.
## The removed chunk = chink

def process_content6():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
                                    ## Removing from the chink one or more verb, preposition, determiner, the words "to"
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()

    except Exception as e:
        print(str(e))

#process_content6()

#%% 7 - Named Entity Recognition

## Two options:
##     False (default) - Recognize named entities as their respective type (people, place, location)
##     True - Recognize all named entities

def process_content7():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
    except Exception as e:
        print(str(e))


#process_content7()

"""
NE Type and Examples (binary=False (default))
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
"""

#%% 8 - Lemmatizing: More powerful than stemming
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemlist = ["cats", "cacti", "geese", "rocks", "python", "better"]

for word in lemlist:
    print(lemmatizer.lemmatize(word))
## Default is lemmatize as noun: pos="n"    
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("better", pos="n"))

#%% 9 - NLTK Corpora: Can download online, however best is to nlkt.download() everything

## To find where something is:
import nltk
#print(nltk.__file__)

## C:\Users\HP\AppData\Roaming\nltk_data\corpora (very big data set)


from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

#print(tok[5:15])

#%% 10 - WordNet
from nltk.corpus import wordnet

syns = wordnet.synsets("program")

## synset
print(syns[0].name())

## just the word
print(syns[0].lemmas()[0].name())

## definition of word
print(syns[0].definition())

## examples
print(syns[0].examples())

synonyms = []

synonyms1 = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

#print(set(synonyms))
#print(set(antonyms))

## Wu and Palmer method for semantic related-ness (compares similiarity of
## two words and their tenses)

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

#print(w1.wup_similarity(w2))

w3 = wordnet.synset('cat.n.01')
w4 = wordnet.synset('car.n.01')
#print(w1.wup_similarity(w3))
#print(w1.wup_similarity(w4))

#%% 11 - Text Classification

## Goal can be broad. You could try to classify text as politics or military.
## You could try to classify by gender of author. A popular task is to identify
## a body of text as spam or not spam (email filters).

## We, on the other hand, will try to create a sentiment analysis algorithm.
## Let's try it.
# 
import nltk
import random
from nltk.corpus import movie_reviews

## In each category (pos or neg), take all of the file IDs, 
## store the word_tokenized version for the file ID, 
## followed by the positive or negative label in one big list.
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

## Shuffle it up for training and testing 
random.shuffle(documents)
#print(documents[0])

## Collect all words in reviews
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

## Perform frequency distribution to find out the most common words.
all_words = nltk.FreqDist(all_words)


## Prints what is says, though including punctuation.
#print(all_words.most_common(15)) 

## Prints how often a word appears.
#print(all_words["stupid"]) 

#%% 12 - Converting words to Features
import nltk
import random
from nltk.corpus import movie_reviews

## See chapter 11
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

## Contains top 3000 most common words
word_features = list(all_words.keys())[:3000]

## Next, build a function that will find these top 3000 words in our positive
## and negative documents, marking them as either positive or negative
## according to whether they are in the text or not.

## SELFNOTE: How about keeping tracking of how many times it repeats?

def find_features(document):
    words= set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

## We can print one feature as so:
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

## Do for all our documents, saving the feature existence booleans and respective pos neg category
## Note how documents is already randomized.
featuresets = [(find_features(rev), category) for (rev, category) in documents]

#%% 13 - Naive Bayes Classifier

## Uses supervised learning: training_set and testing_set

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

## Training classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy:", (nltk.classify.accuracy(classifier, testing_set)))

#classifier.show_most_informative_features(15)

#%% 14 - Saving Classifiers

## Time to actually save our classifier. We do this through Pickle.
import pickle

## Opens up a pickle file, preparing to write in bytes "wb".
## We are saving as naivebayes, but could equally say "lol.pickle"

save_classifier = open("naivebayes.pickle", "wb")

## Then we use pickle.dump() to dump the data.
pickle.dump(classifier, save_classifier)

## Close file.
save_classifier.close()

## Pickled or serialized object saved in script directory.

## How do we open and use the classifier?

## The .pickle file is a serialized object. Now we read it into memory.
## Open file to read as bytes "rb".
classifier_f = open("naivebayes.pickle", "rb")

## Load the file into memory. Save data to classifier variable.
classifier = pickle.load(classifier_f)

## Close file.
classifier_f.close()

#%% 15 - Scikit-Learn (sklearn)

## NLTK people realized the importance of sklearn, so created a SklearnClassifier
## API (of sorts).

from nltk.classify.scikitlearn import SklearnClassifier

## Now you can use any sklearn classifier. Let's bring variations of Naive Bayes algos (GaussianNB fails)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy:", nltk.classify.accuracy(BNB_classifier, testing_set))

## Let us bring more!
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)))

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)))

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set)))

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)))

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)))

#%% 16 - Combining Algorithms

## We'll combine the algorithms with votes! (Choose the best)
## We want our new classifier to act like a typical NLTK classifier.

## Import NTLK's classifier class
from nltk.classify import ClassifierI

## Mode will be used to choose the most popular vote.
from statistics import mode

## Our class will inherit from the NLTK classifier class
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        ## Assign list of classifiers that are passed to our class
        self._classifiers = classifiers
      
    ## Since nltk uses classify, we will write it as well
    def classify(self, features):
        votes = []
        ## Iterate through list of classifier objects
        for c in self._classifiers:
                ## Classify based on features
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

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set)))

for i in range(6):
    print("Classification:", voted_classifier.classify(testing_set[i][0]), 
          "Confidence:",voted_classifier.confidence(testing_set[i][0]))

## @TODO CHECK WHAT VOTING ACTUALLY MEANS

#%% 17 - Investigating Bias

"""
The most major issue is that we have a fairly biased algorithm. You can test 
this yourself by commenting-out the shuffling of the documents, then training 
against the first 1900, and leaving the last 100 (all positive) reviews. 
Test, and you will find you have very poor accuracy.

Conversely, you can test against the first 100 data sets, all negative, and 
train against the following 1900. You will find very high accuracy here. 
This is a bad sign. It could mean a lot of things, and there are many options 
for us to fix it.

That said, the project I have in mind for us suggests we go ahead and use a 
different data set anyways, so we will do that. In the end, we will find this 
new data set still contains some bias, and that is that it picks up negative 
things more often. The reason for this is that negative reviews tend to be 
"more negative" than positive reviews are positive. Handling this can be done 
with some simple weighting, but it can also get complex fast. 
Maybe a tutorial for another day. For now, we're going to just grab a new 
dataset, which we'll be discussing in the next tutorial.
"""

#%% 18 - Improving Training Data for sentiment analysis


## We need a new methodology for creating our "documents" variable, and then we
## also need a new way to create the "all_words" variable.

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

## Need to adjust our feature finding function, tokenizing by word in the doc.

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
	
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)







