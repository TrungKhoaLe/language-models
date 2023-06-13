from typing import List, Dict
import numpy as np
from .process_tweet import process_tweet


def build_freqs(tweets: List[str], ys: np.ndarray) -> Dict:
    yslist = np.squeeze(ys).tolist()

    freqs = {}

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)

            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs
