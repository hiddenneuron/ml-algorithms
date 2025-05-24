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
