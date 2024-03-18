from sklearn.calibration import cross_val_predict
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

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

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print("\n")

def classifier_performance_measurements(classifier, x_train, y_train, x_test, y_test, multilable=False):
    classifier.fit(x_train, y_train)
    
    cvs = cross_val_score(classifier, x_train, y_train, cv=3, scoring="accuracy")
    print("Cross validation scores:")
    display_scores(cvs)

    y_train_predict = cross_val_predict(classifier, x_train, y_train, cv=3)
    print("Confusion matrix:")
    if multilable:
        print(confusion_matrix(y_train, y_train_predict))
    else:
        print(multilabel_confusion_matrix(y_train, y_train_predict))

    print("Precision score:")
    if(multilable):
        print(precision_score(y_train, y_train_predict, average="micro"))
    else:
        print(precision_score(y_train, y_train_predict, average="macro"))
    
    print("Recall score:")
    if(multilable):
        print(recall_score(y_train, y_train_predict, average="micro"))
    else:
        print(recall_score(y_train, y_train_predict, average="macro"))



sgd_clf = SGDClassifier(random_state=42)

classifier_performance_measurements(sgd_clf, x_train, y_train, x_test, y_test, True)

# x, y = mnist["data"], mnist["target"]

# x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# digit = x_train.iloc[0].to_numpy()

# digit_image = digit.reshape(28, 28)

# plt.imshow(digit_image, cmap="binary")
# plt.axis("off")
# plt.show()

