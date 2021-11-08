################################################################################
# FILE: mnist_data.py
# WRITER : Aviel Shtern
# LOGIN : aviel.shtern
# ID: 206260499
# EXERCISE : Introduction to Machine Learning: Exercise 3 - Classification 2021
################################################################################


from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from time import time

MODELS_NAME_Q14 = ["logistic", "soft_svm", "k_neighbors", "tree"]

ACCURACY_PLACE = 0
TIME_TO_FIT = 1
TIME_TO_TEST = 2

NUM_SAMPLES = [50, 100, 300, 500]
NUM_OF_ITER_Q14 = 50

# load the data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# we need only two classes (1,0)
train_images = np.logical_or((train_y == 0), (train_y == 1))
test_images = np.logical_or((test_y == 0), (test_y == 1))
x_train, y_train = train_X[train_images], train_y[train_images]
x_test, y_test = test_X[test_images], test_y[test_images]


def Q12():
    """Play with the dataset - Draw 3 images of samples labeled with ’0’ and 3 images of samples labeled with ’1’"""
    image_0 = x_train[y_train == 0]
    image_1 = x_train[y_train == 1]
    indexes_0 = np.random.randint(image_0.shape[0], size=3)
    indexes_1 = np.random.randint(image_0.shape[0], size=3)
    for index in indexes_0:
        plt.imshow(image_0[index])
        plt.show()
    for index in indexes_1:
        plt.imshow(image_1[index])
        plt.show()


def rearrange_data(X):
    """given data as a array of size nˆ28ˆ28, returns a new matrix of size n ˆ 784 with the same data"""
    return X.reshape(-1, 784)


def draw_samples(m):
    """Draw m training points (x1,...,xm) by choosing uniformly and at random from the train set.
     Note: The training data should always have points from two classes"""
    indexes = np.random.randint(x_train.shape[0], size=m)
    cond = np.sum(y_train[indexes])
    while cond == m or cond == 0:
        indexes = np.random.randint(x_train.shape[0], size=m)
    return rearrange_data(x_train[indexes, :, :]), y_train[indexes]


def Q14():
    """Compares performance Classificaton of two digits from the MNIST Dataset MNIST Dataset"""
    info_accuracy_and_time = {"logistic": ([0] * len(NUM_SAMPLES), [0] * len(NUM_SAMPLES), [0] * len(NUM_SAMPLES)),
                              "tree": ([0] * len(NUM_SAMPLES), [0] * len(NUM_SAMPLES), [0] * len(NUM_SAMPLES)),
                              "k_neighbors": ([0] * len(NUM_SAMPLES), [0] * len(NUM_SAMPLES), [0] * len(NUM_SAMPLES)),
                              "soft_svm": ([0] * len(NUM_SAMPLES), [0] * len(NUM_SAMPLES), [0] * len(NUM_SAMPLES))}

    for i in range(NUM_OF_ITER_Q14):

        for m in range(len(NUM_SAMPLES)):

            MODELS = {"logistic": LogisticRegression(), "tree": DecisionTreeClassifier(max_depth=5),
                      "k_neighbors": KNeighborsClassifier(n_neighbors=1), "soft_svm": SVC(kernel='linear')}
            X_train, Y_train = draw_samples(NUM_SAMPLES[m])
            X_test, Y_test = rearrange_data(x_test), y_test

            for model in MODELS_NAME_Q14:
                cur_model = MODELS[model]

                time_before_fit = time()
                cur_model.fit(X_train, Y_train)
                info_accuracy_and_time[model][TIME_TO_FIT][m] += (time() - time_before_fit) / NUM_OF_ITER_Q14

                time_before_test = time()
                info_accuracy_and_time[model][ACCURACY_PLACE][m] += (cur_model.score(X_test, Y_test) / NUM_OF_ITER_Q14)
                info_accuracy_and_time[model][TIME_TO_TEST][m] += (time() - time_before_test)

    plot_Q14(info_accuracy_and_time)


def plot_Q14(info_accuracy_and_time):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for model in MODELS_NAME_Q14:
        ax.plot(NUM_SAMPLES, info_accuracy_and_time[model][ACCURACY_PLACE], label=model)
    ax.set_xlabel("num of sumpels (m)")
    ax.set_ylabel("accuracy")
    ax.set_title("Q14")
    ax.legend()
    fig.show()

    for model in MODELS_NAME_Q14:
        print(f"times for: {model}")
        for m in range(len(NUM_SAMPLES)):
            time_to_fit_for_m = info_accuracy_and_time[model][TIME_TO_FIT][m]
            time_to_test_for_m = info_accuracy_and_time[model][TIME_TO_TEST][m]
            print(f"{NUM_SAMPLES[m]} sumples: fit - {time_to_fit_for_m}, prdict - {time_to_test_for_m}")
