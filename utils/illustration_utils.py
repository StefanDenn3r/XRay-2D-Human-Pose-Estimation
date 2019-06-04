import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d  # although it is show


def draw_terrain(output):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    X_shape, Y_shape = output.shape
    X, Y = np.meshgrid(np.arange(X_shape), np.arange(Y_shape))
    heights = output
    ax.plot_surface(X, Y, heights, cmap=plt.get_cmap('jet'))
    ax.set_zlim(-1, 1)
    plt.show()


def draw_heatmap(output):
    X_shape, Y_shape = output.shape
    fig, ax = plt.subplot(121)
    plt.imshow(output)
    plt.show()
