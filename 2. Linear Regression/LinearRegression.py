


# Linear Regression
'''
To understand the pattern or the slope of a given dataset

Estimation: the dataset has a linear pattern
Slope: y = mx + b
To know error we use MSE

To find the best fitting line we need to find the value of our parameters, and to do that we need to find the derivative
parameters of our model so weight and the bias that will give us the minimum mean squared error and to do that we
need to calculate the derivative or the gradient of mean squared error.

Optimization Technique: Gradient Descent
Learning rate: How fast or slow to go in the direction that gradient descent tells us to go

We need a good learning rate. if your learning rate is too low it might cause a problem of you not achieving the minimum
error because you're approaching it very slowly if your learning rate is too high you might keep jumping around in your
airspace and you might never be able to find the minimum and that's why it's important to choose a good learning rate.

Steps:

I. Training Phase:
The goal is to find the optimal values for the weight (w) and bias (b) that minimize the error (cost function).
    1) Initialization:

        Initialize the weight (w) to zero (or small random values). If there are multiple features, w will be a vector of zeros.

        Initialize the bias (b) to zero.

    2) Iterative Optimization (Gradient Descent Loop for n_iters iterations):
    For each iteration:

        a. Predict Results: For all data points X in the training set, calculate the predicted output ŷ (y_hat or y_pred) using the current w and b:
        ŷ = wX + b
        (The video mentions doing this efficiently with matrix operations: Y_pred = w * X_matrix + b)

        b. Calculate Error (Cost): Compute the error, typically Mean SquaredError (MSE), between the predicted values ŷ and the actual values y:
        MSE = J(w,b) = (1/N) * Σ(yᵢ - (wxᵢ + b))²
        (where N is the number of data points, yᵢ is the actual value, and (wxᵢ + b) is the predicted value ŷᵢ for the i-th data point).

        c. Calculate Gradients: Determine the gradients (partial derivatives) of the MSE with respect to w (denoted as dw) and b (denoted as db). These indicate the direction to adjust w and b to reduce the error.
        Simplified formulas from the video (after derivation of MSE):

            dw = (1/N) * Σ 2xᵢ(ŷᵢ - yᵢ)

            db = (1/N) * Σ 2(ŷᵢ - yᵢ)
            (The video shows efficient matrix forms for these as well: dw = (1/N) * Xᵀ ⋅ (Y_pred - Y_actual) and db = (1/N) * sum(Y_pred - Y_actual) after absorbing the 2xᵢ factor or just 2)

        d. Update Parameters: Adjust w and b in the opposite direction of their respective gradients, scaled by a learning rate (lr or α):

            w = w - lr * dw

            b = b - lr * db

        e. Repeat: Go back to step 2.a for the specified number of iterations (n_iters).

II. Testing / Prediction Phase (after training):
Once the model is trained and optimal w and b are found:

    Input New Data: Take a new, unseen data point (or set of data points) X_new.

    Predict: Use the final, trained values of w and b to calculate the prediction:
    ŷ_new = wX_new + b
'''

import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, noIter=1000):
        self.lr = lr
        self.noOfIterations = noIter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        noOfSamples, noOfFeatures = X.shape
        self.weights = np.zeros(noOfFeatures)
        self.bias = 0

        for i in range(self.noOfIterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / noOfSamples) * np.dot(X.T, (y_pred - y))
            db = (1 / noOfSamples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
