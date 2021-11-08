################################################################################
# FILE: models.py
# WRITER : Aviel Shtern
# LOGIN : aviel.shtern
# ID: 206260499
# EXERCISE : Introduction to Machine Learning: Exercise 3 - Classification 2021
################################################################################

import numpy as np
import math
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# labels in this exercise +- 1
POSITIVE = 1
NEGATIVE = -1


class Perceptron:

    def __init__(self):
        self.__model = None

    def fit(self, X, y):
        """the perceptron algorithm"""
        X_with_one = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)  # for B (we not assume b = 0)
        w = np.zeros(X_with_one.shape[1])
        cond = ((X_with_one @ w) * y) <= 0
        while np.any(cond):
            idx_for_add = np.nonzero(cond)[0][0]
            w = w + (X_with_one[idx_for_add] * y[idx_for_add])
            cond = ((X_with_one @ w) * y) <= 0
        self.__model = w

    def predict(self, X):
        X_with_one = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return np.sign(X_with_one @ self.__model)

    def get_model(self):
        return self.__model

    def score(self, X, y):
        return score(self, X, y)


class LDA:
    def __init__(self):
        self.inv_cov_mul_mu_positive = None  # I compute anything in a formula that does not include x
        # (i.e. fixed immediately after training)
        self.inv_cov_mul_mu_negative = None
        self.__added_delta_positive = None
        self.__added_delta_negative = None

    def fit(self, X, y):
        """The implementation of LDA according to the conclusions in questions 2 and 3, in "fit" we compute anything in
         a formula that does not include x (i.e. fixed immediately after training)"""
        y = y.reshape(-1, 1)
        num_y_positive = np.sum(y == POSITIVE)
        num_y_negative = y.shape[0] - num_y_positive

        pr_y_positive = num_y_positive / y.shape[0]
        pr_y_negative = num_y_negative / y.shape[0]

        X_positive = X[(y == POSITIVE).flatten()]
        X_negative = X[(y == NEGATIVE).flatten()]

        mu_y_positive = np.sum(X_positive, axis=0) / num_y_positive
        mu_y_negative = np.sum(X_negative, axis=0) / num_y_negative

        X_positive_centered = X_positive - mu_y_positive
        X_negative_centered = X_negative - mu_y_negative

        estimate_cov = ((X_positive_centered.T @ X_positive_centered) / max(1, num_y_positive - 1)
                        + (X_negative_centered.T @ X_negative_centered) / max(1, num_y_negative - 1)) / 2

        inv_cov = np.linalg.inv(estimate_cov)

        self.inv_cov_mul_mu_positive = inv_cov @ mu_y_positive.reshape(-1, 1)
        self.inv_cov_mul_mu_negative = inv_cov @ mu_y_negative.reshape(-1, 1)

        self.__added_delta_positive = -0.5 * (mu_y_positive.reshape(1, -1) @ self.inv_cov_mul_mu_positive) + math.log(
            pr_y_positive)
        self.__added_delta_negative = -0.5 * (mu_y_negative.reshape(1, -1) @ self.inv_cov_mul_mu_negative) + math.log(
            pr_y_negative)

    def predict(self, X):
        delta_positive = (X @ self.inv_cov_mul_mu_positive) + self.__added_delta_positive
        delta_negative = (X @ self.inv_cov_mul_mu_negative) + self.__added_delta_negative
        return np.sign(delta_positive - delta_negative)

    def score(self, X, y):
        return score(self, X, y)


class SVM:
    def __init__(self):
        self.__svc = SVC(C=1e10, kernel='linear')  # Hard_svm
        self.__model = None

    def fit(self, X, y):
        self.__model = self.__svc.fit(X, y)

    def predict(self, X):
        return self.__model.predict(X)

    def get_model(self):
        return self.__model

    def score(self, X, y):
        return score(self, X, y)


class Logistic:
    def __init__(self):
        self.__logistic = LogisticRegression(solver='liblinear')
        self.__model = None

    def fit(self, X, y):
        self.__model = self.__logistic.fit(X, y)

    def predict(self, X):
        return self.__model.predict(X)

    def score(self, X, y):
        return score(self, X, y)


class DecisionTree:
    def __init__(self):
        self.__tree = DecisionTreeClassifier(max_depth=3)
        self.__model = None

    def fit(self, X, y):
        self.__model = self.__tree.fit(X, y)

    def predict(self, X):
        return self.__model.predict(X)

    def score(self, X, y):
        return score(self, X, y)


def score(model, X, y):
    """return a dictionary with: num_samples, error (error rate), accuracy, FPR,TPR, precision, and specificty
     These values are calculated according to the formulas we saw in class"""
    y_head = model.predict(X)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(y.shape[0]):
        predict = y_head[i]
        actualy = y[i]
        if predict == actualy == POSITIVE:
            tp += 1
        if predict == POSITIVE and predict != actualy:
            fp += 1
        if predict == actualy == NEGATIVE:
            tn += 1
        if predict == NEGATIVE and predict != actualy:
            fn += 1
    num_negative = tn + fp
    num_positive = tp + fn
    num_samples = num_negative + num_positive
    return {"num_samples": num_samples, "error": (fp + fn) / num_samples, "accuracy": (tp + tn) / num_samples,
            "FTR": fp / num_negative, "TPR": tp / num_positive, "precision": tp / max((tp + fp), 1),
            "specificty": tn / num_negative}
