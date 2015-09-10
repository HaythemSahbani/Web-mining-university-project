import pickle
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords

class Polarity_classifier:

    def __init__(self):
        pass

    def bigram_word_feats(self, words, score_fn=BigramAssocMeasures.chi_sq, n=200):
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
        return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

    def bag_of_word_feats(self, words):
        return dict([(word, True) for word in words])

    def stopword_filtered_word_feats(self, words):
        stopset = set(stopwords.words('english'))
        return dict([(word, True) for word in words if word not in stopset])

    def set_polarity_bigram_classifier(self, json_tweets):
        f = open('bigram_classifier.pickle')
        classifier = pickle.load(f)
        f.close()
        for tweet in json_tweets:
            tweet["polarity"] = classifier.classify(self.bigram_word_feats(tweet["text"].split()))

    def set_polarity_bag_classifier(self, json_tweets):
        f = open('bag_classifier.pickle')
        classifier = pickle.load(f)
        f.close()
        for tweet in json_tweets:
            tweet["polarity"] = classifier.classify(self.bag_of_word_feats(tweet["text"].split()))

    def set_polarity_stop_classifier(self, json_tweets):
        f = open('stop_word_classifier.pickle')
        classifier = pickle.load(f)
        f.close()
        for tweet in json_tweets:
            tweet["polarity"] = classifier.classify(self.stopword_filtered_word_feats(tweet["text"].split()))
