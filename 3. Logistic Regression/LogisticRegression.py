'''
Logistic Regression
To classify data points based on their features — typically binary classification (0 or 1).

Estimation: The dataset has a pattern that can be separated by a sigmoid-shaped curve.
Instead of predicting a continuous value, logistic regression predicts a probability between 0 and 1.

Formula:
    y = sigmoid(wx + b)
    where sigmoid(z) = 1 / (1 + e^(-z))

To determine the "error" or how good the predictions are, we use Log Loss (Binary Cross-Entropy).

To find the best fitting parameters (w, b), we minimize the loss using Gradient Descent, just like in linear regression.

Optimization Technique: Gradient Descent
Learning rate: Controls how large or small the steps are in the direction of the negative gradient (just like linear regression).

A good learning rate is crucial — too low: slow convergence; too high: overshooting the minimum loss.

---

## Steps:

### I. Training Phase:
The goal is to find the optimal values for the weight (w) and bias (b) that minimize the Binary Cross-Entropy loss.

1) Initialization:
- Initialize the weight (w) to 0 (or small random values).
- Initialize the bias (b) to 0.

2) Iterative Optimization (Gradient Descent Loop for `n_iters` iterations):

For each iteration:

a. Predict Probabilities:
For all data points X in the training set, calculate the predicted probability (ŷ or y_pred):
z = wX + b

ŷ = sigmoid(z) = 1 / (1 + e^(-z))

b. Compute Loss (Binary Cross-Entropy or Log Loss):
Loss (J(w,b)) = - (1/N) * Σ [yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ)]

This penalizes incorrect confident predictions more harshly.

c. Calculate Gradients:
Partial derivatives of the loss w.r.t w and b:
dw = (1/N) * Xᵀ ⋅ (ŷ - y)

db = (1/N) * Σ(ŷ - y)
(ŷ - y) is the prediction error vector.

d. Update Parameters:
Adjust w and b in the opposite direction of the gradients:
w = w - lr * dw

b = b - lr * db

e. Repeat:
Loop through steps a to d for the specified number of iterations.

---

II. Testing / Prediction Phase (after training):

Input New Data:
Take a new, unseen data point (or dataset) X_new

Predict Probability:
ŷ_new = sigmoid(wX_new + b)


Predict Class Label (if required):
Threshold the probability:
If ŷ_new >= 0.5 → predict 1

If ŷ_new < 0.5  → predict 0

'''
import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return (1/(1+np.exp(-x)))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return  class_pred

