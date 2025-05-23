import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.title("Generated Regression Dataset")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

reg = LinearRegression(lr=0.01, noIter=1000)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_value = mse(y_test, predictions)
print(mse_value)

y_line = reg.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, color="blue", label="Training Data", alpha=0.6)
plt.scatter(X_test[:, 0], y_test, color="green", label="Test Data", alpha=0.6)
plt.plot(X[:, 0], y_line, color="red", linewidth=2, label="Regression Line") 
plt.title("Linear Regression Fit")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# print(mse)
# y_line = reg.predict(X)
#
# plt.scatter(X[:, 0], y, color="b", label="Data")
# plt.plot(X[:, 0], y_line, color="r", label="Regression Line")
# plt.legend()
# plt.show()
