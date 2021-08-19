""" Retrieve tweets and users then create embeddings and populate the DB. """
import tweepy
import spacy
from .models import db, Tweet, User
from os import getenv


TWITTER_API_KEY = getenv('TWITTER_API_KEY')
TWITTER_API_KEY_SECRET = getenv('TWITTER_API_KEY_SECRET')
TWITTER_AUTH = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_KEY_SECRET)
TWITTER = tweepy.API(TWITTER_AUTH)

# Load NLP Model
nlp = spacy.load('my_nlp_model')
def vectorize_tweet(tweet_text):
    return nlp(tweet_text).vector

def add_or_update_user(username):
    try:
        print(username)
        twitter_user = TWITTER.get_user(username)
        db_user = (User.query.get(twitter_user.id)) or User(
            id=twitter_user.id, name=username
        )
        db.session.add(db_user)

        tweets = twitter_user.timeline(
            count=200, exclude_replies=True, include_rts=False,
            tweet_mode='Extended'
        )

        for tweet in tweets:
            vectorized_tweet = vectorize_tweet(tweet.text)
            db_tweet = Tweet(id=tweet.id, text=tweet.text,
                             vect=vectorized_tweet)
            db_user.tweets.append(db_tweet)
            db.session.add(db_tweet)

        db.session.commit()
    except Exception as e:
        print('Error processing{}: {}'.format(username, e))
        raise e


def insert_example_users():
    """We will get an error if we run this twice without dropping & creating"""
    add_or_update_user('elonmusk')
    add_or_update_user('ben')

def prediction_model(name1, name2, text):
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    user1 = TWITTER.get_user(name1)
    user2 = TWITTER.get_user(name2)
    user1_tweets = user1.timeline(count=200, exclude_replies=True,
                                  include_rts=False, tweet_mode='Extended')
    user2_tweets = user2.timeline(count=200, exclude_replies=True,
                                  include_rts=False, tweet_mode='Extended')
    user1_tweet_vecs = [vectorize_tweet(tweet.text) for tweet in user1_tweets]
    user2_tweet_vecs = [vectorize_tweet(tweet.text) for tweet in user2_tweets]
    user1_user2_vecs = np.vstack([user1_tweet_vecs, user2_tweet_vecs])
    user1_user2_labels = np.concatenate(
        (np.zeros(len(user1_tweet_vecs)),
        np.ones(len(user2_tweet_vecs))), axis=0)
    log_reg = LogisticRegression().fit(user1_user2_vecs, user1_user2_labels)

    new_text = vectorize_tweet(text).reshape(1, -1)
    return log_reg.predict(new_text)[0]



