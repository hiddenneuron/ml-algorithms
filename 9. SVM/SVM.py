'''
Support Vector Machine (SVM)

SVM is a powerful supervised learning algorithm used for classification (and also regression). It works by finding the optimal boundary (hyperplane) that best separates data points of different classes.

Estimation: The data can be separated by a hyperplane with a maximum margin (distance) between classes.

Goal: To find the hyperplane that maximizes the margin between the two classes. The points that lie closest to the hyperplane are called support vectors.

Steps:

I. Training Phase:

1. Input the labeled training data.

2. Find the Optimal Hyperplane:

   * A hyperplane is a line in 2D, a plane in 3D, and a flat surface in higher dimensions.
   * The SVM tries to find the hyperplane that separates the classes with the widest margin.
   * Margin = distance between the hyperplane and the nearest data points from each class.
   * These nearest points are the “support vectors” and are critical in defining the decision boundary.

3. Solve Optimization Problem:

   * The algorithm solves a mathematical optimization problem to maximize the margin.
   * For hard margin SVM: assumes perfect separation with no error.
   * For soft margin SVM: allows some misclassification but tries to minimize it using a penalty parameter (C), which controls the trade-off between margin size and classification error.

4. Use Kernel Trick (if needed):

   * If the data is not linearly separable, map it to a higher-dimensional space using a kernel function (e.g., linear, polynomial, radial basis function).
   * This allows SVM to find a linear separation in that higher-dimensional space.

II. Prediction Phase:

1. Input a new data point.

2. Compute which side of the hyperplane the point lies on using the learned weights and bias (or kernel function if used).

3. Assign the class based on the sign of the output:

   * If (w · x + b) > 0, classify as class 1
   * If (w · x + b) < 0, classify as class 0

Key Points:

* Effective in high-dimensional spaces.
* Works well even when number of dimensions > number of samples.
* Memory-efficient because it only uses support vectors.
* Choice of kernel and parameter C affects performance.

'''

import numpy as np

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# Testing(copied directly fromm github)
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    clf = SVM()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    print("SVM classification accuracy", accuracy(y_test, predictions))

    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()