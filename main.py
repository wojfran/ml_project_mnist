from matplotlib import pyplot as plt
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from performance import classifier_performance_measurements
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import threading

def shift_images(data, dy, dx):
    return np.array([shift(image.reshape(28, 28), [dy, dx]).reshape(784) for image in data])
    
def artificialAugmentation(data, target):
    left = shift_images(data, 0, -1)
    right = shift_images(data, 0, 1)
    up = shift_images(data, -1, 0)
    down = shift_images(data, 1, 0)

    augmented_data = np.concatenate((data, left, right, up, down))
    augmented_target = np.concatenate((target, target, target, target, target))

    shuffle_index = np.random.permutation(len(augmented_data))
    augmented_data = augmented_data[shuffle_index]
    augmented_target = augmented_target[shuffle_index]

    return augmented_data, augmented_target

mnist = fetch_openml('mnist_784', version=1)

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

sgd_clf = SGDClassifier(random_state=42, max_iter=50, tol=1e-3)
k_neighbors_clf = KNeighborsClassifier(n_neighbors=5)
random_forest_clf = RandomForestClassifier()

t1 = threading.Thread(target=classifier_performance_measurements, args=("RF_base", random_forest_clf, x_train, y_train, x_test, y_test, True, False))
t2 = threading.Thread(target=classifier_performance_measurements, args=("RF_aug", random_forest_clf, aug_x_train, aug_y_train, x_test, y_test, True, False))
t3 = threading.Thread(target=classifier_performance_measurements, args=("SGD_base", sgd_clf, x_train, y_train, x_test, y_test, True, False))
t4 = threading.Thread(target=classifier_performance_measurements, args=("SGD_aug", sgd_clf, aug_x_train, aug_y_train, x_test, y_test, True, False))
t5 = threading.Thread(target=classifier_performance_measurements, args=("KNC_base", k_neighbors_clf, x_train, y_train, x_test, y_test, True, False))
t6 = threading.Thread(target=classifier_performance_measurements, args=("KNC_aug", k_neighbors_clf, aug_x_train, aug_y_train, x_test, y_test, True, False))

print("Random Forest Classifier - not augmented")
t1.start()
print("Random Forest Classifier - augmented")
t2.start()
print("SGD Classifier - not augmented")
t3.start()
print("SGD Classifier - augmented")
t4.start()
print("K Neighbors Classifier - not augmented")
t5.start()
print("K Neighbors Classifier - augmented")
t6.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()

print("All threads finished")
