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


def PCA(x):
    """Perform Principal Component Analysis (PCA) on the dataset x.

    Parameters
    ----------
    x : ndarray
        Input data of shape (n_samples, n_features).
    """

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
