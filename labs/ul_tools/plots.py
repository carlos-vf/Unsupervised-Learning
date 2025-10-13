import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_figure(x, c, title="", colorbar_title="", cmap='viridis', dynamic=False):
    """
    Plot 2D or 3D data.

    Parameters
    ----------
    x : ndarray
        Data points to plot. Shape should be (n_samples, 2) for 2D or (n_samples, 3) for 3D.
    c : ndarray
        Color values for each point. Shape should be (n_samples,).
    title : str, optional
        Title of the plot. Default is "".
    colorbar_title : str, optional
        Title for the colorbar. Default is "".
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    dynamic : bool, optional
        If True and data is 3D, use an interactive plot. Default is False.
    """

    # 2d plot
    if x.shape[1] == 2:
        y = x[:, 1]
        x = x[:, 0]

        fig = plt.figure(figsize=(8,6))
        plt.scatter(x, y, c=c, cmap=cmap)

        # Labels
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)

        plt.show()

    # 3d plot
    elif x.shape[1] == 3:

        y = x[:, 1]
        z = x[:, 2]
        x = x[:, 0]

        # Dynamic plot
        if dynamic:
            fig = go.Figure(data=[go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=10,
                    color=c,
                    colorscale=cmap,
                    opacity=1,
                    colorbar=dict(title=colorbar_title)
                )
            )])

            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="X"),
                    yaxis=dict(title="Y"),
                    zaxis=dict(title="Z")
                ),
                title=title,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            fig.show()

        # Static plot
        else:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x, y, z, c=c, cmap=cmap)
            plt.colorbar(sc, ax=ax, label=colorbar_title)

            # Labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)

            plt.show()



def pca_scree_plot(eigenvalues):
    """Plot the eigenvalue spectrum (scree plot) from PCA.

    Parameters
    ----------
    eigenvalues : ndarray
        Array of eigenvalues from PCA, sorted in descending order.
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(eigenvalues) + 1), eigenvalues, 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue (log-scale)')
    plt.title('Eigenvalue Spectrum (Scree Plot)')
    plt.grid(True)
    plt.xticks(range(1, len(eigenvalues) + 1))
    plt.show()



def project_2d(X, pc, color, class_names=None):
    """Project data onto the first two principal components and plot.
    
    Parameters
    ----------
    X : DataFrame
        Input data of shape (n_samples, n_features).
    pc : ndarray
        Principal components from PCA of shape (n_features, n_features).
    color : ndarray
        Color values for each point. Shape should be (n_samples,).
    class_names : list, optional
        List of class names for legend. Default is None.
    """

    W_2D = pc[:, :2]
    X_train_pca_2D = X.dot(W_2D)

    # 2D Projection Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_train_pca_2D.iloc[:, 0], X_train_pca_2D.iloc[:, 1], c=color, cmap='viridis', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Training Data Projected onto First Two Principal Components')
    if class_names is not None:
        plt.legend(handles=scatter.legend_elements()[0], labels=class_names.tolist())
    plt.grid(True)
    plt.show()



def project_3d(X, pc, color, class_names=None):
    """Project data onto the first three principal components and plot.

    Parameters
    ----------
    X : DataFrame
        Input data of shape (n_samples, n_features).
    pc : ndarray
        Principal components from PCA of shape (n_features, n_features).
    color : ndarray
        Color values for each point. Shape should be (n_samples,).
    class_names : list, optional
        List of class names for legend. Default is None.

    """
    W_3D = pc[:, :3]
    X_train_pca_3D = X.dot(W_3D)

    # 3D Projection Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter_3d = ax.scatter(X_train_pca_3D.iloc[:, 0], X_train_pca_3D.iloc[:, 1], X_train_pca_3D.iloc[:, 2], c=color, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('Training Data Projected onto First Three Principal Components')
    if class_names is not None:
        ax.legend(handles=scatter_3d.legend_elements()[0], labels=class_names.tolist())
    plt.show()



def plot_isomap(y, color, title):
    """Plot 2D Isomap embedding.
    Parameters
    ----------
    y : ndarray
        2D embedded data of shape (n_samples, 2).
    color : ndarray
        Color values for each point. Shape should be (n_samples,).
    title : str
        Title of the plot.
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(y[:, 0], y[:, 1], c=color, edgecolor='k', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid()
    plt.axis('equal')
    plt.show()
    