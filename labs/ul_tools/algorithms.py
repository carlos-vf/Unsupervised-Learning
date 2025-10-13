import numpy as np


def scale_and_center(x):
    return (x - x.mean()) / x.std()


def PCA():
    # Covariance matrix
    c = np.cov(x, rowvar=False)
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(c)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

