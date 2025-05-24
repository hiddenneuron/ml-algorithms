'''
Principal Component Analysis (PCA):

PCA is a dimensionality reduction technique used to simplify a dataset by reducing the number of features while retaining the most important information.
Estimation: The dataset has correlated features and can be represented in fewer dimensions without losing much information.

Goal: To project high-dimensional data into a lower-dimensional space with minimal information loss by identifying new axes (called principal components) that capture the maximum variance in the data.

Steps:

I. Preprocessing:

1. Mean Normalization:
   * For each feature in the dataset, subtract the mean of that feature from all its values. This centers the data around the origin.
   * This step is important because PCA is sensitive to the scale and mean of the data.

2. Compute the Covariance Matrix:
   * The covariance matrix captures the relationships (correlations) between different features in the dataset.
   * For a dataset with m features, this will be an m x m matrix.

II. Compute Principal Components:

1. Eigen Decomposition:
   * Calculate the eigenvectors and eigenvalues of the covariance matrix.
   * Eigenvectors represent the directions (axes) of maximum variance.
   * Eigenvalues represent the amount of variance captured by each eigenvector.

2. Sort Eigenvectors:
   * Sort the eigenvectors based on their corresponding eigenvalues in descending order.
   * The eigenvector with the highest eigenvalue is the first principal component and captures the most variance.

III. Dimensionality Reduction:

1. Choose Top K Components:
   * Select the top k eigenvectors (principal components) based on the desired amount of variance to retain (e.g., 95% of the total variance).

2. Project Data:
   * Multiply the original normalized dataset with the top k eigenvectors.
   * This transforms the data into the new lower-dimensional space.

IV. Reconstruction (Optional):

* You can approximate the original data using the reduced dimensions by reversing the transformation, although some information may be lost.

Application:
* PCA is commonly used for data visualization, noise reduction, feature extraction, and improving model performance by removing redundant features.
'''

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering:
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance, functions needs samples as columns
        cov = np.cov(X.T)

        # eigenvectores and eigenvalues
        # use eigh since covariance matrix is symmetric
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        # eigenvectors v =[:,i] coumns vector, transpose this for easier calculations
        self.components = eigenvectors[:, :self.n_components].T

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)

# Testing(copied directly fromm github)
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # data = datasets.load_digits()
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
