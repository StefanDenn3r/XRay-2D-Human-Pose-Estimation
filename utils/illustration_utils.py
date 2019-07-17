import matplotlib.pyplot as plt
import numpy as np
import random
import mpl_toolkits.mplot3d.axes3d as axes3d  # although it is show
from itertools import combinations_with_replacement

colors = list(combinations_with_replacement(np.arange(0.1, 1, 0.2), 3))
random.seed(3)
random.shuffle(colors)
colors = colors[6:29]

def draw_terrain(output):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    X_shape, Y_shape = output.shape
    X, Y = np.meshgrid(np.arange(X_shape), np.arange(Y_shape))
    heights = output
    ax.plot_surface(X, Y, heights, cmap=plt.get_cmap('jet'))
    plt.show()


def draw_red_landmark(array, x, y, radius):
    radius += 3
    array[0, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 1
    array[1, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 0
    array[2, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 0


def draw_green_landmark(array, x, y, radius):
    radius += 3
    array[0, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 0
    array[1, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 1
    array[2, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = 0

def draw_colored_landmark(array, x, y, radius, color):
    array[0, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = colors[color][0]
    array[1, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = colors[color][1]
    array[2, (y - radius):(y + radius + 1), (x - radius):(x + radius + 1)] = colors[color][2]