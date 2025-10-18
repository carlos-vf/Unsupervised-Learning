import matplotlib.pyplot as plt
import numpy as np
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



def plot_eigenvalue_spectrum(eigenvalues):
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



def project_principal_components(x, y, z=None, dynamic=False, labels=None, title=None, color=None, cmap='viridis', class_names=None):
    """Project data onto principal components and plot.
    Parameters
    ----------
    x : ndarray
        First principal component values. Shape should be (n_samples,).
    y : ndarray
        Second principal component values. Shape should be (n_samples,).
    z : ndarray, optional
        Third principal component values. Shape should be (n_samples,). Default is None.
    dynamic : bool, optional
        If True and z is provided, create an interactive 3D plot. Default is False.
    labels : ndarray, optional
        Discrete class labels for each point. Shape should be (n_samples,).
        This will be used for coloring if provided.
    title : str, optional
        Title of the plot. Default is None.
    color : ndarray, optional
        Continuous color values for each point. Shape should be (n_samples,).
        Used only if 'labels' is None.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    class_names : list or array, optional
        List of class names for legend. Used only if 'labels' is not None.
    """

    # 2D Projection Plot
    if z is None:
        plt.figure(figsize=(8, 6))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            # Generate a set of colors from the specified colormap
            colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
            
            for i, ul in enumerate(unique_labels):
                idx = (labels == ul)
                # Use class_names for the legend if provided
                label = class_names[i] if (class_names is not None and i < len(class_names)) else f'Class {ul}'
                plt.scatter(x[idx], y[idx], color=colors[i], label=label, alpha=0.7)
            
            plt.legend()
        else:
            # Plot with continuous color (e.g., from 'phi')
            scatter = plt.scatter(x, y, c=color, cmap=cmap, alpha=0.7)
            plt.colorbar(scatter, label='Color Value')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        if title is not None:
            plt.title(title)
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    # 3D Projection Plot
    elif z is not None:
        if dynamic:
            fig = go.Figure()

            if labels is not None:
                unique_labels = np.unique(labels)
                # Generate a set of colors from the specified colormap
                colors_mpl = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
                
                for i, ul in enumerate(unique_labels):
                    idx = (labels == ul)
                    # Use class_names for the legend if provided
                    label = class_names[i] if (class_names is not None and i < len(class_names)) else f'Class {ul}'
                    r, g, b, a = colors_mpl[i]
                    color_string = f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})'
                    fig.add_trace(go.Scatter3d(
                        x=x[idx],
                        y=y[idx],
                        z=z[idx],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=color_string,
                            opacity=0.8
                        ),
                        name=str(label)
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color,
                        colorscale=cmap,
                        opacity=0.8,
                        colorbar=dict(title='Color Value')
                    )
                ))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="Principal Component 1"),
                    yaxis=dict(title="Principal Component 2"),
                    zaxis=dict(title="Principal Component 3")
                ),
                title=title if title is not None else "",
                margin=dict(l=0, r=0, b=0, t=40)
            )
            fig.show()

        else:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            if labels is not None:
                unique_labels = np.unique(labels)
                # Generate a set of colors from the specified colormap
                colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
                
                for i, ul in enumerate(unique_labels):
                    idx = (labels == ul)
                    # Use class_names for the legend if provided
                    label = class_names[i] if (class_names is not None and i < len(class_names)) else f'Class {ul}'
                    ax.scatter(x[idx], y[idx], z[idx], color=colors[i], label=label, alpha=0.7)
                
                ax.legend()
            else:
                # Plot with continuous color (e.g., from 'phi')
                scatter = ax.scatter(x, y, z, c=color, cmap=cmap, alpha=0.7)
                fig.colorbar(scatter, ax=ax, label='Color Value')
            
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            if title is not None:
                ax.set_title(title)
            plt.show()


    