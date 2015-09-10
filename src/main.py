from Preprocess import Preprocess
from Preprocess import TweetTokenizer
from twitter_crawl import TwitterCrawl
from Clustering import Clustering
from hashtag_classification import HtagClassifier, plot
from plot import present_all_data
from Polarity_classifier import Polarity_classifier

screen_name = "BarackObama"
# "SenJohnMcCain"
file = screen_name + "_tweets.json"

twt = TwitterCrawl()


##################  Get list of tweets  ################

#option 1: crawl data from twitter and save it into json file
json_tweets = twt.get_feed(screen_name)
twt.save_tweets(file)

#option 2: get data from a file ()
#json_tweets = twt.load_tweets(file)

print("number of tweets : %d" % len(json_tweets))



##########################     Polarity     ######################
pc = Polarity_classifier()

#Choose one of the three polarity classifiers
pc.set_polarity_bag_classifier(json_tweets)
#pc.set_polarity_bigram_classifier(json_tweets)
#pc.set_polarity_stop_classifier(json_tweets)



#################  Preprocess for clustering   ####################
#get text of each tweet
tweet_text = twt.get_tweet_text(json_tweets)
for tweet in tweet_text:
    tweet_text[tweet_text.index(tweet)] = " ".join(
                                                Preprocess().remove_stopwords(
                                                    TweetTokenizer().tokenize(
                                                        Preprocess().expand_contraction(tweet))))




########################   Clustering     #########################

#####    k-means   ######
clr = Clustering()

best_k = clr.gap_statistic(tweet_text, kmin=2, kmax=10)
clr.best_kmeans(best_k, tweet_text)
clr.set_tweet_topic(json_tweets)
twt.json_tweets = json_tweets
twt.save_tweets(file)

#present topic popularity evolution and its polarity evolution over the time
present_all_data(best_k, json_tweets)


##### Htag clussifier ###
Ht = HtagClassifier()
plot(json_tweets)
