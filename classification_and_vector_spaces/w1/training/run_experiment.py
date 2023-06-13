import numpy as np
from data import prepare_data
from data.utils import build_freqs
from model import train_algorithms
from model import activations
from typing import List, Dict, Tuple


def predict_tweet(tweet: List[str], freqs: Dict, theta: np.ndarray) -> float:
    x = prepare_data.extract_features(tweet, freqs)
    y_pred = activations.sigmoid(x @ theta)

    return y_pred


def test_logistic_regression(test_X: List[str], test_y: np.ndarray,
                             freqs: Dict, theta: np.ndarray) -> float:
    y_hat = []

    for tweet in test_X:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)
    accuracy = (y_hat == np.squeeze(test_y)).sum() / len(test_X)

    return accuracy


def main() -> Tuple[float, np.ndarray]:
    pos_tweets, neg_tweets = prepare_data.load_downloaded_tweet_samples()
    train_X, train_y, test_X, test_y = prepare_data.form_train_test_data(pos_tweets,
                                                                         neg_tweets)
    freqs = build_freqs.build_freqs(train_X, train_y)

    X = np.zeros((len(train_X), 3))
    Y = train_y
    
    for i in range(len(train_X)):
        X[i, :] = prepare_data.extract_features(train_X[i], freqs)

    J, theta = train_algorithms.gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

    acc_train = test_logistic_regression(train_X, train_y, freqs, theta)
    acc_test = test_logistic_regression(test_X, test_y, freqs, theta)

    print(f"[INFO] Train accuracy: {acc_train}, train loss: {J}")
    print(f"[INFO] Test accuracy: {acc_test}")
    
    return J, theta


if __name__ == "__main__":
    _, theta = main()
    # save the weights
    np.save("chkpts/theta.npy", theta)

