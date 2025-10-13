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

    