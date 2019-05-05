import torch


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def jointmseloss(self, output, target):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
    loss = 0

    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx].squeeze()
        heatmap_gt = heatmaps_gt[idx].squeeze()
        loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

    return loss / num_joints
