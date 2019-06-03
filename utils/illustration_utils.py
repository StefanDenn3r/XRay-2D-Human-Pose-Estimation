import matplotlib.pyplot as plt
import numpy as np


def draw_terrain(output):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    X_shape, Y_shape = output.shape
    X, Y = np.meshgrid(np.arange(X_shape), np.arange(Y_shape))
    heights = output
    ax.plot_surface(X, Y, heights, cmap=plt.get_cmap('jet'))
    ax.set_zlim(-1, 1)
    plt.show()
