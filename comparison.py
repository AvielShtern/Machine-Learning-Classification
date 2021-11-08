################################################################################
# FILE: comparison.py
# WRITER : Aviel Shtern
# LOGIN : aviel.shtern
# ID: 206260499
# EXERCISE : Introduction to Machine Learning: Exercise 3 - Classification 2021
################################################################################


import models
import matplotlib.pyplot as plt
import numpy as np

NUM_OF_ITER = 500
NUM_SANPLE_TEST = 10000  # k
NUM_SUMPLES = [5, 10, 15, 25, 70]  # m's


def draw_points(m):
    """given an integer m returns a pair X, y where X is matrix (dim = (m,2)) where each column represents an i.i.d
    sample from the normal distribution, and y (+-1) is its corresponding label"""
    mu = np.array([0, 0])
    cov = np.eye(2)
    x = np.random.multivariate_normal(mu, cov, m)
    y = np.sign((x @ np.array([[0.3], [-0.5]])) + 0.1)  # f(x)
    while abs(np.sum(y)) == m:  # The training data should always have points from two classes
        x = np.random.multivariate_normal(mu, cov, m)
        y = np.sign((x @ np.array([[0.3], [-0.5]])) + 0.1)
    return x, y.flatten()


def Q9():
    """plots for m in {5,10,15,25,70}"""
    for m in NUM_SUMPLES:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        X, y = draw_points(m)

        perceptron = models.Perceptron()
        svm = models.SVM()
        svm.fit(X, y)
        perceptron.fit(X, y)

        ax.scatter(X[y == models.POSITIVE, 0], X[y == models.POSITIVE, 1], color='red', label='POSITIVE')
        ax.scatter(X[y == models.NEGATIVE, 0], X[y == models.NEGATIVE, 1], color='black', label='NEGATIVE')
        ax.plot([-4, 4], [-4 * perceptron.get_model()[0] / -perceptron.get_model()[1] + perceptron.get_model()[2] /
                          - perceptron.get_model()[1],
                          4 * perceptron.get_model()[0] / -perceptron.get_model()[1] + perceptron.get_model()[2] / -
                          perceptron.get_model()[1]], label='Perceptron')
        ax.plot([-4, 4],
                [-4 * svm.get_model().coef_[0, 0] / -svm.get_model().coef_[0, 1] + svm.get_model().intercept_ / -
                svm.get_model().coef_[0, 1],
                 4 * svm.get_model().coef_[0, 0] / -svm.get_model().coef_[0, 1] + svm.get_model().intercept_ / -
                 svm.get_model().coef_[0, 1]], label='SVM')
        ax.plot([-4, 4], [-4 * 0.3 / 0.5 + 0.1 / 0.5, 4 * 0.3 / 0.5 + 0.1 / 0.5], label='True Hypothesis')

        ax.set_title(f'Q9: for m={m}')
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        fig.show()


def Q10():
    """compare the accuracy of three algorithms (Perceptron, SVM, LDA)"""
    accuracy_perceptron = []
    accuracy_svm = []
    accuracy_lda = []

    for m in NUM_SUMPLES:
        curr_accuracy_perceptron = 0
        curr_accuracy_svm = 0
        curr_accuracy_lda = 0
        for i in range(NUM_OF_ITER):
            X, y = draw_points(m)
            X_test, y_test = draw_points(NUM_SANPLE_TEST)

            perceptron = models.Perceptron()
            svm = models.SVM()
            lda = models.LDA()

            perceptron.fit(X, y)
            svm.fit(X, y)
            lda.fit(X, y)

            curr_accuracy_perceptron += (perceptron.score(X_test, y_test)["accuracy"]) / NUM_OF_ITER
            curr_accuracy_svm += (svm.score(X_test, y_test)["accuracy"]) / NUM_OF_ITER
            curr_accuracy_lda += (lda.score(X_test, y_test)["accuracy"]) / NUM_OF_ITER

        accuracy_perceptron.append(curr_accuracy_perceptron)
        accuracy_svm.append(curr_accuracy_svm)
        accuracy_lda.append(curr_accuracy_lda)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(NUM_SUMPLES, accuracy_svm, label='SVM')
    ax.plot(NUM_SUMPLES, accuracy_perceptron, label='Perceptron')
    ax.plot(NUM_SUMPLES, accuracy_lda, label='LDA')
    ax.set_xlabel("num of sumpels (m)")
    ax.set_ylabel("accuracy")
    ax.set_title("Q10")
    ax.legend()
    fig.show()
