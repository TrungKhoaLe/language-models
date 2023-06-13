from typing import List, Tuple, Dict
import nltk
from os import getcwd
from .utils.process_tweet import process_tweet
from .utils.build_freqs import build_freqs
import numpy as np
import pandas as pd
nltk.download("twitter_samples")
nltk.download("stopwords")
from nltk.corpus import twitter_samples


def load_downloaded_tweet_samples() -> Tuple[List[str], List[str]]:
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    return all_positive_tweets, all_negative_tweets

def form_train_test_data(pos_tweets: List[str],
                         neg_tweets: List[str]) -> Tuple[List[str], np.ndarray,
                                                         List[str], np.ndarray]:
    test_pos = pos_tweets[4000:]
    train_pos = pos_tweets[:4000]
    test_neg = neg_tweets[4000:]
    train_neg = neg_tweets[:4000]

    train_x = train_pos + train_neg
    test_x = test_pos + test_neg
    train_y = np.append(np.ones((len(train_pos), 1)),
                        np.zeros((len(train_neg), 1)), axis=0)
    test_y = np.append(np.ones((len(test_pos), 1)),
                       np.zeros((len(test_neg), 1)), axis=0)

    return train_x, train_y, test_x, test_y

def extract_features(tweet: List[str], freqs: Dict) -> np.ndarray:
    
    word_l = process_tweet(tweet)
    # three elements for [bias, positive, negative] counts
    x = np.zeros(3)
    # set the bias term to 1
    x[0] = 1

    for word in word_l:
        # for the positive tweets
        x[1] += freqs.get((word, 1.0), 0)
        # for the negative tweets
        x[2] += freqs.get((word, 0.0), 0)

    # add the batch dimension for further processing
    x = x[None, :]

    return x


if __name__ == "__main__":
    all_positive_tweets, all_negative_tweets = load_downloaded_tweet_samples()
    train_X, train_y, test_X, test_y = form_train_test_data(all_positive_tweets,
                                                            all_negative_tweets)
    assert len(all_positive_tweets) == 5000
    assert train_y.shape == (8000, 1)
    freqs = build_freqs(train_X, train_y)
    print("type(freqs) = " + str(type(freqs)))
    print("len(freqs) = " + str(len(freqs.keys())))
    print('This is an example of a positive tweet: \n', train_X[0])
    print('\nThis is an example of the processed version of the tweet: \n',
            process_tweet(train_X[0]))
