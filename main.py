from matplotlib import pyplot as plt
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from performance import classifier_performance_measurements
import numpy as np

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

def artificialAugmentation(data, target):
    left = shift(data, [-1, 0], cval=0, mode="constant")
    right = shift(data, [1, 0], cval=0, mode="constant")
    up = shift(data, [0, -1], cval=0, mode="constant")
    down = shift(data, [0, 1], cval=0, mode="constant")

    augmented_data = np.concatenate((data, left, right, up, down))
    augmented_target = np.concatenate((target, target, target, target, target))

    shuffle_index = np.random.permutation(len(augmented_data))
    augmented_data = augmented_data[shuffle_index]
    augmented_target = augmented_target[shuffle_index]

    return augmented_data, augmented_target


mnist = fetch_openml('mnist_784', version=1)

shifted = shift_image((mnist.data.to_numpy())[0], 5, 1)

data = pd.concat([pd.DataFrame(mnist.data), pd.DataFrame(mnist.target)], axis=1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["class"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

x_train = strat_train_set.drop(columns=["class"])
y_train = strat_train_set["class"]

x_test = strat_test_set.drop(columns=["class"])
y_test = strat_test_set["class"]

aug_x_train, aug_y_train = artificialAugmentation(x_train.to_numpy(), y_train.to_numpy())

sgd_clf = SGDClassifier(random_state=42)

# classifier_performance_measurements(sgd_clf, x_train, y_train, x_test, y_test, True, False)
classifier_performance_measurements(sgd_clf, aug_x_train, aug_y_train, x_test, y_test, True, False)

