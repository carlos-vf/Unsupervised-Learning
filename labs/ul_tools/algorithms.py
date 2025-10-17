import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def center(x):
    """Center the dataset x.

    Parameters
    ----------
    x : ndarray
        Input data of shape (n_samples, n_features).
    """
    return x - x.mean()

def scale(x):
    """Scale the dataset x to have unit variance.

    Parameters
    ----------
    x : ndarray
        Input data of shape (n_samples, n_features).
    """
    return x / x.std()


def PCA(x, n_components=None):
    """Perform Principal Component Analysis (PCA) on the dataset x.

    Parameters
    ----------
    x : ndarray
        Input data of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to return. If None, return all components. Default is None.
    """

    # Center the data
    x = center(x)

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

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    # Compute the projections
    projections = x @ eigenvectors

    return projections, eigenvalues, eigenvectors
    
    


def dijkstra(g, source):
    """Dijkstra's algorithm to find the shortest paths from a source node in a weighted graph.

    Parameters
    ----------
    g : networkx.Graph
        A weighted graph where edges have a 'weight' attribute.
    source : node
        The starting node for the shortest path calculation.
    """

    dist = {v: np.inf for v in g.nodes}
    dist[source] = 0
    visited = set()
    queue = set(g.nodes)

    while queue:
        u = min((v for v in queue), key=lambda v: dist[v])
        queue.remove(u)
        visited.add(u)

        for neighbor, attr in g[u].items():
            if neighbor in queue:
                alt = dist[u] + attr['weight']
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
    return dist



def geosdesic_distance(g):
    """Compute the geodesic distance matrix for a weighted graph using Dijkstra's algorithm.
    
    Parameters
    ----------
    g : networkx.Graph
        A weighted graph where edges have a 'weight' attribute.
    """
    nodes = sorted(list(g.nodes))
    delta = np.zeros((len(nodes), len(nodes)))

    for i, source in enumerate(nodes):
        dist_dict = dijkstra(g, source)
        
        for j in range(i, len(nodes)):
            target = nodes[j]
            
            if target in dist_dict:
                delta[i, j] = dist_dict[target]
                delta[j, i] = dist_dict[target]
            else:
                delta[i, j] = np.inf
                delta[j, i] = np.inf

    return delta



def isomap(x, n_components, n_neighbors=5):
    """Perform Isomap dimensionality reduction on the dataset x.
    Parameters
    ----------
    x : ndarray
        Input data of shape (n_samples, n_features).
    n_components : int
        Number of dimensions to reduce to.
    n_neighbors : int, optional
        Number of neighbors to consider for each point. Default is 5.
    """
    
    # Nearest Neighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(x)
    neigh_dist, neigh_idx = neigh.kneighbors()
    
    # Neighbor Graph
    g = nx.Graph()
    for i in range(x.shape[0]):
        for j in range(len(neigh_idx[i])):
            neighbor_index = neigh_idx[i][j]
            distance = neigh_dist[i][j]
            g.add_edge(i, neighbor_index, weight=distance)
    
    if not nx.is_connected(g):
        raise ValueError(
            f"Graph is not connected with n_neighbors={n_neighbors}. "
            "Please increase n_neighbors."
        )

    # Distance matrix (by Dijkstra)
    delta = geosdesic_distance(g)

    # Double centering
    delta_squared = delta ** 2
    row_avg = np.mean(delta_squared, axis=1)
    col_avg = np.mean(delta_squared, axis=0)
    total_avg = np.mean(delta_squared)
    gram_matrix = -0.5 * (delta_squared - row_avg[:, np.newaxis] - col_avg[np.newaxis, :] + total_avg)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Final coordinates
    top_eigenvalues = eigenvalues[:n_components]
    top_eigenvectors = eigenvectors[:, :n_components]
    
    # Ensure no negative eigenvalues
    positive_eigenvalues = np.maximum(top_eigenvalues, 0)
    
    # Coordinates = Eigenvectors * sqrt(Eigenvalues)
    embedded_coordinates = top_eigenvectors @ np.diag(np.sqrt(positive_eigenvalues))

    return embedded_coordinates



def kernel_PCA(x, kernel_func, degree=None, sigma=None, n_components=None):
    """Perform Kernel Principal Component Analysis (Kernel PCA) on the dataset x.
    Parameters
    ----------
    x : ndarray
        Input data of shape (n_samples, n_features).
    kernel_func : str
        Kernel function to use: 'linear', 'polynomial', or 'gaussian'.
    degree : int, optional
        Degree for the polynomial kernel. Required if kernel_func is 'polynomial'.
    sigma : float, optional
        Standard deviation for the Gaussian kernel. Required if kernel_func is 'gaussian'.
    n_components : int, optional
        Number of principal components to return. If None, return all components. Default is None.
    """

    # Kernel matrix computation
    if kernel_func == 'polynomial':
        if degree is None:
            raise ValueError("For polynomial kernel 'degree' parameter must be provided.")
        k = (x @ x.T + 1) ** degree

    elif kernel_func == 'gaussian':
        if sigma is None:
            raise ValueError("For Gaussian kernel 'sigma' parameter must be provided.")
        k = np.exp(-np.linalg.norm(x[:, np.newaxis] - x[np.newaxis, :], axis=2)**2 / (2 * sigma**2))
    
    elif kernel_func == 'linear':
        k = x @ x.T
    
    else:
        raise ValueError("Unsupported kernel function. Use 'linear', 'polynomial' or 'gaussian'.")

    # Centering the Kernel matrix
    n_samples = k.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples # n x n matrix with all entries equal to 1/n
    k_centered = k - one_n @ k - k @ one_n + one_n @ k @ one_n

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(k_centered)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        
    # Compute the projections
    projections = eigenvectors * np.sqrt(eigenvalues)

    return projections, eigenvalues, eigenvectors
