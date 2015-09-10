
import collections
import nltk.metrics
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
neg_ids = movie_reviews.fileids('neg')
pos_ids = movie_reviews.fileids('pos')



def bag_of_word_feats(words):
    return dict([(word, True) for word in words])


neg_feats = [(bag_of_word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in neg_ids]
pos_feats = [(bag_of_word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in pos_ids]



neg_limit = len(neg_feats)*3/4
pos_limit = len(pos_feats)*3/4


trainfeats = neg_feats[:neg_limit] + pos_feats[:pos_limit]
testfeats = neg_feats[neg_limit:] + pos_feats[pos_limit:]
print 'train on %d instances, test.ipynb on %d instances' % (len(trainfeats), len(testfeats))
print neg_feats[1]

classifier = NaiveBayesClassifier.train(trainfeats)
import pickle
f = open('bag_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
print 'neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg'])
