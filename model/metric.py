import numpy as np
from torch import nn

from utils import util

threshold = 0.05


def smooth_l1_loss(output, target):
    # Uses Huber Loss
    return util.apply_loss(nn.SmoothL1Loss(), output, target)


def l1_loss(output, target):
    return util.apply_loss(nn.L1Loss(reduction='mean'), output, target)


def logistic_loss(output, target):
    return util.apply_loss(nn.SoftMarginLoss(reduction='mean'), output, target)


def binary_cross_entropy(output, target):
    return util.apply_loss(nn.BCELoss(reduction='mean'), output, target)


def percentage_correct_keypoints(output, target):

    predictions = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    distance_threshold = np.linalg.norm(target.shape[2:4]) * 0.1  # 10% of image diagonal as distance_threshold
    target_landmarks_batch = [[np.unravel_index(np.argmax(i_target[idx], axis=None), i_target[idx].shape)
                               for idx in range(i_target.shape[0])] for i_target in target]
    true_positives = 0
    all_predictions = 0
    for prediction in predictions:
        pred_landmarks_batch = [[np.unravel_index(np.argmax(i_output[idx], axis=None), i_output[idx].shape) for idx in
                                 range(i_output.shape[0])] for i_output in prediction]

        for idx, (pred_landmarks, target_landmarks) in enumerate(zip(np.array(pred_landmarks_batch), np.array(target_landmarks_batch))):
            for channel_idx, (pred_landmark, target_landmark) in enumerate(zip(pred_landmarks, target_landmarks)):
                # check if either landmark is correctly predicted as not given OR predicted landmarks is within radius
                if (abs(prediction[idx, channel_idx, pred_landmark[0], pred_landmark[1]]) <= threshold and np.sum(target_landmark) == 0
                        or np.linalg.norm(pred_landmark - target_landmark) <= distance_threshold):
                    true_positives += 1

                all_predictions += 1

    return (true_positives / all_predictions) * 100


def keypoint_distance_loss(output, target):
    predictions = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    target_landmarks_batch = [[np.unravel_index(np.argmax(i_target[idx], axis=None), i_target[idx].shape)
                               for idx in range(i_target.shape[0])] for i_target in target]
    sum_distance = 0.0
    all_predictions = 0
    for prediction in predictions:
        pred_landmarks_batch = [[np.unravel_index(np.argmax(i_output[idx], axis=None), i_output[idx].shape) for idx in
                                 range(i_output.shape[0])] for i_output in prediction]

        for idx, (pred_landmarks, target_landmarks) in enumerate(zip(np.array(pred_landmarks_batch), np.array(target_landmarks_batch))):
            for channel_idx, (pred_landmark, target_landmark) in enumerate(zip(pred_landmarks, target_landmarks)):
                if np.sum(target_landmark) == 0:
                    continue
                sum_distance += np.linalg.norm(pred_landmark - target_landmark)
                #prediction[idx, channel_idx, target_landmark[0], target_landmark[1]], prediction[idx, channel_idx, pred_landmark[0], pred_landmark[1]], target[idx, channel_idx, target_landmark[0], target_landmark[1]]
                all_predictions += 1

    return sum_distance / all_predictions
