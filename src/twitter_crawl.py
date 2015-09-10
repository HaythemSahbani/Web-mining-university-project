import json
import twitter

class TwitterCrawl:

    consumer_key = ""
    consumer_secret = ""
    access_token_key = ""
    access_token_secret = ""

    def __init__(self, consumer_key=consumer_key,
                 consumer_secret=consumer_secret,
                 access_token_key=access_token_key,
                 access_token_secret=access_token_secret):

        self.api = twitter.Api(consumer_key=consumer_key,
                               consumer_secret=consumer_secret,
                               access_token_key=access_token_key,
                               access_token_secret=access_token_secret)

    def get_feed(self, screen_name):
        #getting the last 200 tweets
        tweets = self.api.GetUserTimeline(screen_name=screen_name, count=200)
        #extend the tweets dataset to 200*20 tweets
        for i in range(0, 20):
            tweets += self.api.GetUserTimeline(screen_name=screen_name, count=200, max_id=tweets[-1].id)
        self.json_tweets = [{"id": tweet.id, "text": tweet.text, "topic": "", "date": tweet.created_at, "polarity": ""} for tweet in tweets]
        return self.json_tweets

    @staticmethod
    def get_tweet_text(json_frame):
        tweet_list = list()
        for tweet in json_frame:
            text = tweet['text']
            tweet_list.append(text.encode('ascii', errors='ignore'))
        return tweet_list

    def save_tweets(self, file):
        f = open(file, "w+")
        f.write(json.dumps(self.json_tweets))
        f.close()

    def load_tweets(self, file):
        f = open(file, "r")
        self.json_tweets = json.load(f)
        f.close()
        return self.json_tweets
