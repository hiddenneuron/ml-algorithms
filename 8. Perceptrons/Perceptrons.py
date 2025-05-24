'''
Perceptron

The perceptron is one of the simplest types of artificial neural networks used for binary classification. It models a single neuron that makes decisions by weighing input features.

Estimation: The data is linearly separable, meaning a straight line (or hyperplane) can divide the classes.

Goal: To find a set of weights and a bias that can correctly classify the input data into two classes.

Model:
Output = 1 if (w · x + b) ≥ 0
Output = 0 otherwise
Where w is the weight vector, x is the input vector, and b is the bias.

Steps:

I. Training Phase:

1. Initialization:

   * Set all weights and the bias to zero (or small random values).
   * Choose a learning rate (a small positive number).

2. Iterative Update:

   * For each training sample:
     a) Calculate the predicted output using:
     y\_pred = 1 if (w · x + b) ≥ 0, else 0
     b) Compare with actual label y.
     c) If prediction is wrong, update the weights and bias:

     * w = w + learning\_rate \* (y - y\_pred) \* x
     * b = b + learning\_rate \* (y - y\_pred)
   * Repeat for multiple passes (epochs) over the training data until all points are correctly classified or a maximum number of iterations is reached.

II. Prediction Phase:

1. Input a new sample x\_new.
2. Compute:
   y\_pred = 1 if (w · x\_new + b) ≥ 0, else 0
3. Output the predicted class.

Key Points:

* Only works if the data is linearly separable.
* Fast and simple to implement.
* Forms the foundation for more complex models like multilayer perceptrons (MLPs).

'''

import numpy as np


def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0 , 1, 0)

        # learn weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


# Testing (copied directly from github)
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()