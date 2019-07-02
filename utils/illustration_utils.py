import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d  # although it is show


def draw_terrain(output):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    X_shape, Y_shape = output.shape
    X, Y = np.meshgrid(np.arange(X_shape), np.arange(Y_shape))
    heights = output
    ax.plot_surface(X, Y, heights, cmap=plt.get_cmap('jet'))
    plt.show()


def draw_red_landmark(array, x, y, radius):
    array[0, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 1
    array[1, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 0
    array[2, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 0