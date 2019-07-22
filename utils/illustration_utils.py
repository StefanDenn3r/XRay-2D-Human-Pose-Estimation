import random
import mpl_toolkits.mplot3d.axes3d as axes3d  # although it is show
from itertools import combinations_with_replacement

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np

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


def draw_images_on_tensorboard(writer, config, data, output, sample_idx, target):
    target = target.cpu().detach().numpy()
    outputs = output.cpu().detach().numpy()  # predictions from all stages
    target_landmarks = [[np.unravel_index(np.argmax(i_target[idx], axis=None), i_target[idx].shape)
                         for idx in range(i_target.shape[0])] for i_target in target]
    target_radius = max(1, int(target.shape[-1] * 0.02))
    for idx, image in enumerate(data.cpu().detach().numpy()):
        all_outputs, all_predictions = [], []

        image_target = np.concatenate((np.copy(image), np.copy(image), np.copy(image)))
        image_base = np.copy(image_target)

        image_radius = max(1, int(data.shape[-1] * 0.02))

        for channel_idx, (y, x) in enumerate(target_landmarks[idx]):
            if np.sum(target[idx, channel_idx]) > 0:
                x *= (data.shape[-1] // target.shape[-1])
                y *= (data.shape[-1] // target.shape[-1])
                draw_green_landmark(image_target, x, y, image_radius)
                draw_colored_landmark(image_target, x, y, image_radius, channel_idx)

        all_predictions.append(image_target * 255)

        for stageIdx in range(outputs.shape[0]):
            image_pred = np.copy(image_base)
            output = outputs[stageIdx]
            pred_landmarks = [[np.unravel_index(np.argmax(i_output[idx], axis=None), i_output[idx].shape)
                               for idx in range(i_output.shape[0])] for i_output in output]
            arr = []

            for channel_idx, ((target_y, target_x), (pred_y, pred_x)) in enumerate(zip(target_landmarks[idx], pred_landmarks[idx])):
                temp_target = np.expand_dims(target[idx, channel_idx], axis=0)
                curr_target = np.concatenate((np.copy(temp_target), np.copy(temp_target), np.copy(temp_target)))
                if np.sum(target[idx, channel_idx]) > 0:
                    draw_green_landmark(curr_target, target_x, target_y, target_radius - 2)

                temp_output = np.expand_dims(output[idx, channel_idx], axis=0)
                curr_output = np.concatenate((np.copy(temp_output), np.copy(temp_output), np.copy(temp_output)))

                if temp_output[0, pred_y, pred_x] > config['threshold']:
                    draw_green_landmark(curr_output, pred_x, pred_y, target_radius - 2)
                elif np.sum(target[idx, channel_idx]) > 0:
                    # landmark below threshold but should be present.
                    draw_red_landmark(curr_output, pred_x, pred_y, target_radius - 2)

                arr.append(curr_target)
                arr.append(curr_output)

            all_outputs.append(make_grid(torch.tensor(arr)).cpu().detach().numpy())

            for channel_idx, (y, x) in enumerate(pred_landmarks[idx]):
                if output[idx, channel_idx, y, x] > config['threshold']:
                    x *= (data.shape[-1] // target.shape[-1])
                    y *= (data.shape[-1] // target.shape[-1])
                    draw_green_landmark(image_pred, x, y, image_radius)
                    draw_colored_landmark(image_pred, x, y, image_radius, channel_idx)
                elif np.sum(target[idx, channel_idx]) > 0:
                    # landmark below threshold but should be present.
                    x *= (data.shape[-1] // target.shape[-1])
                    y *= (data.shape[-1] // target.shape[-1])
                    draw_red_landmark(image_pred, x, y, image_radius)
                    draw_colored_landmark(image_pred, x, y, image_radius, channel_idx)

            all_predictions.append(image_pred * 255)

        writer.add_image(f'target_output_{sample_idx}', make_grid(torch.tensor(all_outputs), nrow=outputs.shape[0], padding=5, pad_value=1.0))
        writer.add_image(f'target_predictions_{sample_idx}',
                         make_grid(torch.tensor(all_predictions), nrow=outputs.shape[0] + 1, normalize=True, padding=5, pad_value=1.0))


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
