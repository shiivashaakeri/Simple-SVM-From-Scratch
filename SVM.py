
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import datasets


# Calculating gradient
def calculate_cost_gradient(W, X_batch, Y_batch):
    # We do this so that the function is general for both SGD and steepest descent
    if type(Y_batch) == np.int64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (regularizationParam * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw/len(Y_batch)
    return dw

# Training based on stochastic gradient descent. 
def SGD(XData, YData):
    weights = np.zeros(XData.shape[1])
    for epoch in range(epochs):
        # Shuffle to prevent repeating update cycles and be stochastic
        X, Y = shuffle(XData, YData)
        # Update on each data point that was chosen randomly
        for ind, x in enumerate(X):
            gradient = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (lr * gradient)

    return weights

# Training based on steepest descent
def SD(XData, YData):
    weights = np.zeros(XData.shape[1])
    # For each epoch, we calculate the gradient for all data points
    for epoch in range(epochs):
        gradient = calculate_cost_gradient(weights, XData, YData)
        weights = weights - (lr * gradient)

    return weights

# As we are using one-vs-rest method, we need adjust labels for each models
def splitYData(data, label):
    newData = data.copy()
    newData[newData != label] = -1
    newData[newData == label] = 1
    return newData

def getPredictions(XData, W):
    predictions = np.array([])
    for i in range(XData.shape[0]):
        yp = np.sign(np.dot(XData[i], W))
        predictions = np.append(predictions, yp)
    return predictions

def printConfusionMatrix(W0, W1, W2, data, label):
    matrix = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    for i in range(label.shape[0]):
        y0 = np.sign(np.dot(data[i], W0))
        y1 = np.sign(np.dot(data[i], W1))
        y2 = np.sign(np.dot(data[i], W2))
        
        if label[i] == 0:
            if y0 == 1:
                matrix[0][0] += 1
            if y1 == 1:
                matrix[0][1] += 1
            if y2 == 1:
                matrix[0][2] += 1
        elif label[i] == 1:
            if y0 == 1:
                matrix[1][0] += 1
            if y1 == 1:
                matrix[1][1] += 1
            if y2 == 1:
                matrix[1][2] += 1
        elif label[i] == 2:
            if y0 == 1:
                matrix[2][0] += 1
            if y1 == 1:
                matrix[2][1] += 1
            if y2 == 1:
                matrix[2][2] += 1
        
    print(matrix)

# Hyperparameters
lr = 0.01
epochs = 50000
regularizationParam = 10000

# Ready the data
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

X = np.c_[X, np.ones(X.shape[0])]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

y_train_M0 = splitYData(y_train, 0)
y_train_M1 = splitYData(y_train, 1)
y_train_M2 = splitYData(y_train, 2)

y_test_M0 = splitYData(y_test, 0)
y_test_M1 = splitYData(y_test, 1)
y_test_M2 = splitYData(y_test, 2)

# Train models
W_M0 = SD(X_train.copy(), y_train_M0)
W_M1 = SD(X_train.copy(), y_train_M1)
W_M2 = SD(X_train.copy(), y_train_M2)

# Testing models
y_train_predicted_M0 = getPredictions(X_train, W_M0)
y_test_predicted_M0 = getPredictions(X_test, W_M0)

y_train_predicted_M1 = getPredictions(X_train, W_M1)
y_test_predicted_M1 = getPredictions(X_test, W_M1)

y_train_predicted_M2 = getPredictions(X_train, W_M2)
y_test_predicted_M2 = getPredictions(X_test, W_M2)

print('accuracy on train dataset for class 0: %f' % (accuracy_score(y_train_M0, y_train_predicted_M0)))
print('accuracy on train dataset for class 1: %f' % (accuracy_score(y_train_M1, y_train_predicted_M1)))
print('accuracy on train dataset for class 2: %f' % (accuracy_score(y_train_M2, y_train_predicted_M2)))

print('accuracy on test dataset for class 0: %f' % (accuracy_score(y_test_M0, y_test_predicted_M0)))
print('accuracy on test dataset for class 1: %f' % (accuracy_score(y_test_M1, y_test_predicted_M1)))
print('accuracy on test dataset for class 2: %f' % (accuracy_score(y_test_M2, y_test_predicted_M2)))

printConfusionMatrix(W_M0, W_M1, W_M2, X_test, y_test)

