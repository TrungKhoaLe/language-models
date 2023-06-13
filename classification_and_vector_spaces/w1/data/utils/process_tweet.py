from typing import List
import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


stemmer = PorterStemmer()
stopwords_english = stopwords.words("english")

def process_tweet(tweet: str) -> List[str]:
    # remove stock market tickers such as $GE
    tweet = re.sub(r"\$\w*", "", tweet)
    # remove old style retweet text RT
    # [  ] means one of the characters in the
    # brackets, \s means space, + means one or more
    # words. e.g. RT @lahrose23
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    # remove hyperlinks
    tweet = re.sub(r"https://[^\s\n\r]+", "", tweet)
    # remove hashtags
    tweet = re.sub(r"#", "", tweet)
    # tokenize tweets
    # e.g. @JohnDoe, (including the punctuation and it will be removed)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []

    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean
