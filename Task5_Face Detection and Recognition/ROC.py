from operator import ne
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


# define array of actual values
y_actual = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# define array of predicted values
y_predicted = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 1,
                       0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


def getPerformanceMetrics(TP, FN):
    accuracy = (TP)/(TP+FN)
    recall = (TP)/(TP+FN)  # TPR
    # precision = (TP)/(TP+FP)
    # specificity = (TN)/(TN+FN)
    # FPRate = (FP)/(FP+TN)
    return accuracy


def getConfusionMatrix(act, pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(pred)):
        if pred[i] == act[i]:
            tp += 1
        else:
            fn += 1

    return tp, fn


def ROC(actual, predicted):
    thresholds = np.array(list(range(0, 100, 1)))/100
    ROCXPoints = []
    ROCYPoints = []
    for threshold in thresholds:
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(predicted)):
            if predicted[i] >= threshold:
                predClass = 1
            else:
                predClass = 0

            if predClass == 1 and actual[i] == 1:
                tp += 1
            elif predClass == 0 and actual[i] == 1:
                fn += 1
            elif predClass == 1 and actual[i] == 0:
                fp += 1
            elif predClass == 0 and actual[i] == 0:
                tn += 1
        TPrate = tp/(tp+fn)
        FPRate = fp/(tn+fp)
        # ROC POINTS
        ROCXPoints.append(FPRate)
        ROCYPoints.append(TPrate)
    return ROCXPoints, ROCYPoints


# to get confusion matrix of string lists
# for testing confusion
actualList = ["x", "y", "z", "a", "y", "b", "c", "e", "z", "x"]
predList = ["z", "y", "a", "a", "y", "b", "c", "z", "d", "x"]
tp, fn = getConfusionMatrix(actualList, predList)
print(tp, fn)
accuracy = getPerformanceMetrics(tp, fn)
print(accuracy*100)


# Manual read a data set to test
data = pd.read_csv("auc-case-predictions.csv")
# data for ROC
actual = data["actual"].to_list()
predicted = data["prediction"].to_list()


# to draw ROC
x, y = ROC(actual, predicted)
print(x)
print(y)
plt.scatter(x, y)
plt.plot([0, 1])
plt.show()
