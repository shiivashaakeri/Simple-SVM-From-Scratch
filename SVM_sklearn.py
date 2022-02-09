import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# Two functions to plot data points and SVM decision area
# First we make the mesh grid to have data points from all areas between boundaries
def make_meshgrid(x, y, h=0.02): 
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

# We predict all points to be able to find different sections of space in which SVM outputs different classes
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Load and split data
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create and fit SVM model with linear kernel (based on the information given in the question)
model = svm.SVC(kernel="linear", probability=True)

model.fit(x_train, y_train)


# Plot data points in train dataset and partitioning the input space
fig, sub = plt.subplots()
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = x_train[:, 0], x_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(sub, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
sub.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
sub.set_xlim(xx.min(), xx.max())
sub.set_ylim(yy.min(), yy.max())
sub.set_xlabel("Sepal length")
sub.set_ylabel("Sepal width")
sub.set_xticks(())
sub.set_yticks(())
sub.set_title("SVC with linear kernel")

plt.show()

# Test dataset predict to calculate accuracy, confusion matrix and confidence matrix
pred = model.predict(x_test)
print(confusion_matrix(y_test, pred))

accuracies = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

class_probabilities = model.predict_proba(x_test)

# print(class_probabilities*100)