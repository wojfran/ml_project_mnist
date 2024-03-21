import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import cross_val_predict, label_binarize
from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print("\n")

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(thresholds, precisions[:-1], "b--", label="Precision")
    axs[0].set_xlabel("Threshold")
    axs[0].legend(loc="center left")
    axs[0].set_ylim([0, 1])

    axs[1].plot(thresholds, recalls[:-1], "g-", label="Recall")
    axs[1].set_xlabel("Threshold")
    axs[1].legend(loc="center left")
    axs[1].set_ylim([0, 1])

    plt.show()

def plot_precision_vs_recall(precisions, recalls):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(recalls, precisions, "b-", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.axis([0, 1, 0, 1])

    plt.show()

def plot_roc_curve(fpr, tpr, label=None):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.axis([0, 1, 0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    plt.show()

def multilabel_roc(y_test, labels, y_scores):
    fpr = dict()
    tpr = dict()

    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()

def multilabel_precision_vs_recall(y_test, labels, y_scores):
    precision = dict()
    recall = dict()

    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_scores[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
            
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

def classifier_performance_measurements(filename, classifier, x_train, y_train, x_test, y_test, multilable=False, plot=False):
    cwd = os.getcwd()
    filename = os.path.join(cwd, "performance", filename + ".txt")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        labels = list(set(y_train))

        cvs = cross_val_score(classifier, x_train, y_train, cv=3, scoring="accuracy")
        f.write("Cross validation scores:\n")
        f.write("Scores: " + str(cvs) + "\n")
        f.write("Mean: " + str(cvs.mean()) + "\n")
        f.write("Standard deviation: " + str(cvs.std()) + "\n\n")

        y_train_predict = cross_val_predict(classifier, x_train, y_train, cv=3)

        f.write("Confusion matrix:\n")
        if multilable:
            conf_mx = confusion_matrix(y_train, y_train_predict)
            f.write(str(conf_mx) + "\n")
        else:
            conf_mx = multilabel_confusion_matrix(y_train, y_train_predict)
            f.write(str(conf_mx) + "\n")

        if plot:
            f.write("\nConfusion matrix plot:\n")
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].matshow(conf_mx, cmap=plt.cm.gray)
            axs[0].set_title("Confusion matrix")
            axs[0].axis('off')

            row_sums = conf_mx.sum(axis=1, keepdims=True)
            norm_conf_mx = conf_mx / row_sums
            np.fill_diagonal(norm_conf_mx, 0)
            axs[1].matshow(norm_conf_mx, cmap=plt.cm.gray)
            axs[1].set_title("Confusion matrix errors")
            axs[1].axis('off')

            plt.savefig(filename + "_confusion_matrix.png")
            plt.close()

        f.write("\nScores based on cross-validation on the training set:\n")
        f.write("Precision score:\n")
        f.write(str(precision_score(y_train, y_train_predict, average="macro")) + "\n")
        f.write("Recall score:\n")
        f.write(str(recall_score(y_train, y_train_predict, average="macro")) + "\n")
        f.write("F1 score:\n")
        f.write(str(f1_score(y_train, y_train_predict, average="macro")) + "\n")

        y_test_predict = classifier.fit(x_train, y_train).predict(x_test)
        joblib.dump(classifier, filename + ".joblib")

        f.write("\nScores based on the test set:\n")
        f.write("Precision score:\n")
        f.write(str(precision_score(y_test, y_test_predict, average="macro")) + "\n")
        f.write("Recall score:\n")
        f.write(str(recall_score(y_test, y_test_predict, average="macro")) + "\n")
        f.write("F1 score:\n")
        f.write(str(f1_score(y_test, y_test_predict, average="macro")) + "\n")

        # if multilable and plot:
        #     y_train_binary = label_binarize(y_train, classes=labels)
        #     y_test_binary = label_binarize(y_test, classes=labels)

        #     clf = OneVsRestClassifier(classifier)

        #     clf.fit(x_train, y_train_binary)

        #     y_scores = clf.decision_function(x_test)

        #     multilabel_precision_vs_recall(y_test_binary, labels, y_scores)

        #     multilabel_roc(y_test_binary, labels, y_scores)

        # elif plot:
        #     y_scores = cross_val_predict(classifier, x_train, y_train, cv=3, method="decision_function")
        #     precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
        #     plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        #     plot_precision_vs_recall(precisions, recalls)

        #     fpr, tpr, thresholds = roc_curve(y_train, y_scores)
        #     plot_roc_curve(fpr, tpr)

        #     f.write("\nAUC score:\n")
        #     f.write(str(roc_auc_score(y_train, y_scores)) + "\n")







# The F1
# score favors classifiers that have similar precision and recall. This is not always
# what you want: in some contexts you mostly care about precision, and in other con‐
# texts you really care about recall. For example, if you trained a classifier to detect vid‐
# eos that are safe for kids, you would probably prefer a classifier that rejects many
# good videos (low recall) but keeps only safe ones (high precision), rather than a clas‐
# sifier that has a much higher recall but lets a few really bad videos show up in your
# product (in such cases, you may even want to add a human pipeline to check the clas‐
# sifier’s video selection). On the other hand, suppose you train a classifier to detect
# shoplifters in surveillance images: it is probably fine if your classifier has only 30%
# precision as long as it has 99% recall (sure, the security guards will get a few false
# alerts, but almost all shoplifters will get caught).
