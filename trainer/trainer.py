import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import illustration_utils


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        self.data_loader.dataset.set_sigma()
        print(f'Current sigma: {self.data_loader.dataset.sigma}')
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            target = F.interpolate(target, size=output.shape[-2:])

            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                sample_idx = batch_idx * self.data_loader.batch_size
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    sample_idx,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

                # custom begin

                target = target.cpu().detach().numpy()
                output = output[-1].cpu().detach().numpy()  # only last one relevant for final prediction

                target_landmarks = [[np.unravel_index(np.argmax(i_target[idx], axis=None), i_target[idx].shape)
                                     for idx in range(i_target.shape[0])] for i_target in target]

                pred_landmarks = [[np.unravel_index(np.argmax(i_output[idx], axis=None), i_output[idx].shape)
                                   for idx in range(i_output.shape[0])] for i_output in output]

                target_radius = max(1, int(target.shape[-1] * 0.02))

                for idx, image in enumerate(data.cpu().detach().numpy()):
                    arr = []
                    for channel_idx, ((target_y, target_x), (pred_y, pred_x)) in enumerate(
                            zip(target_landmarks[idx], pred_landmarks[idx])):
                        temp_target = np.expand_dims(target[idx, channel_idx], axis=0)
                        curr_target = np.concatenate((np.copy(temp_target), np.copy(temp_target), np.copy(temp_target)))
                        if np.sum(target[idx, channel_idx]) > 0:
                            illustration_utils.draw_green_landmark(curr_target, target_x, target_y, target_radius)

                        temp_output = np.expand_dims(output[idx, channel_idx], axis=0)
                        curr_output = np.concatenate((np.copy(temp_output), np.copy(temp_output), np.copy(temp_output)))

                        if temp_output[0, pred_y, pred_x] > self.config['threshold']:
                            illustration_utils.draw_green_landmark(curr_output, pred_x, pred_y, target_radius)
                        elif np.sum(target[idx, channel_idx]) > 0:
                            # landmark below threshold but should be present.
                            illustration_utils.draw_red_landmark(curr_output, pred_x, pred_y, target_radius)

                        arr.append(curr_target),
                        arr.append(curr_output),

                    self.writer.add_image(f'target_output_{sample_idx}', make_grid(torch.tensor(arr)))

                    image_target = np.concatenate((np.copy(image), np.copy(image), np.copy(image)))
                    image_pred = np.copy(image_target)

                    image_radius = max(1, int(data.shape[-1] * 0.02))

                    for channel_idx, (y, x) in enumerate(target_landmarks[idx]):
                        if np.sum(target[idx, channel_idx]) > 0:
                            x *= (data.shape[-1] // target.shape[-1])
                            y *= (data.shape[-1] // target.shape[-1])
                            illustration_utils.draw_green_landmark(image_target, x, y, image_radius)

                    for channel_idx, (y, x) in enumerate(pred_landmarks[idx]):
                        if output[idx, channel_idx, y, x] > self.config['threshold']:
                            x *= (data.shape[-1] // target.shape[-1])
                            y *= (data.shape[-1] // target.shape[-1])
                            illustration_utils.draw_green_landmark(image_pred, x, y, image_radius)
                        elif np.sum(target[idx, channel_idx]) > 0:
                            # landmark below threshold but should be present.
                            x *= (data.shape[-1] // target.shape[-1])
                            y *= (data.shape[-1] // target.shape[-1])
                            illustration_utils.draw_red_landmark(image_pred, x, y, image_radius)

                    self.writer.add_image(f'target_predictions_{sample_idx}',
                                          make_grid(torch.tensor([
                                              image_target * 255,
                                              image_pred * 255,
                                          ]), nrow=2, normalize=True))

                # custom end

                if epoch == 1:
                    self.writer.add_image(f'input_{batch_idx}', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                target = F.interpolate(target, size=output.shape[-2:])
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
