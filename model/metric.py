import numpy as np
from scipy.ndimage import gaussian_filter
from torch import nn

from config import CONFIG
from utils import util


def smooth_l1_loss(output, target):
    # Uses Huber Loss
    return util.apply_loss(nn.SmoothL1Loss(), output, target)


def l1_loss(output, target):
    return util.apply_loss(nn.L1Loss(reduction='mean'), output, target)


def mse_loss(output, target):
    return util.apply_loss(nn.MSELoss(reduction='mean'), output, target)


def percentage_correct_keypoints(output, target):
    predictions = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    distance_threshold = 0.03 * np.max(output.shape)  # 15 pixels out of 479 x 615 input image

    target_landmarks_batch = [[np.unravel_index(np.argmax(i_target[idx], axis=None), i_target[idx].shape)
                               for idx in range(i_target.shape[0])] for i_target in target]
    true_positives = 0
    all_predictions = 0
    prediction = predictions[-1]

    pred_landmarks_batch = np.array(
        [gaussian_filter(prediction_channel, sigma=CONFIG['prediction_blur']) for prediction_channel in prediction]
    )

    pred_landmarks_batch = [[np.unravel_index(np.argmax(i_output[idx], axis=None), i_output[idx].shape) for idx in
                             range(i_output.shape[0])] for i_output in pred_landmarks_batch]

    for idx, (pred_landmarks, target_landmarks) in enumerate(
            zip(np.array(pred_landmarks_batch), np.array(target_landmarks_batch))):
        for channel_idx, (pred_landmark, target_landmark) in enumerate(zip(pred_landmarks, target_landmarks)):
            # check if either landmark is correctly predicted as not given OR predicted landmarks is within radius
            if (prediction[idx, channel_idx, pred_landmark[0], pred_landmark[1]] <= CONFIG['threshold']
                    and np.sum(target_landmark) == 0
                    or np.linalg.norm(pred_landmark - target_landmark) <= distance_threshold):
                true_positives += 1

            all_predictions += 1

    return (true_positives / all_predictions) * 100


def keypoint_distance_loss(output, target):
    predictions = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    target_landmarks_batch = [[np.unravel_index(np.argmax(i_target[idx], axis=None), i_target[idx].shape)
                               for idx in range(i_target.shape[0])] for i_target in target]
    pixel_mm = 0.62 * 615 / np.max(output.shape)  # 479 x 615 (image size) : 300 x 384 mm^2 (Detector size)
    sum_distance = 0.0
    all_predictions = 0
    prediction = predictions[-1]

    pred_landmarks_batch = [[np.unravel_index(np.argmax(i_output[idx], axis=None), i_output[idx].shape) for idx in
                             range(i_output.shape[0])] for i_output in prediction]

    pred_landmarks_batch = np.array(
        [gaussian_filter(pred_landmark, sigma=CONFIG['prediction_blur']) for pred_landmark in pred_landmarks_batch])

    for idx, (pred_landmarks, target_landmarks) in enumerate(zip(np.array(pred_landmarks_batch), np.array(target_landmarks_batch))):
        for channel_idx, (pred_landmark, target_landmark) in enumerate(zip(pred_landmarks, target_landmarks)):
            if np.sum(target_landmark) == 0:
                continue
            sum_distance += np.linalg.norm(pred_landmark - target_landmark)
            all_predictions += 1

    return (sum_distance * pixel_mm) / all_predictions
