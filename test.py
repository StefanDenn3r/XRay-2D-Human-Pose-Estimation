import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser, parse_cmd_args


def main(config, resume):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # begin custom
            #

            target_landmarks = [[np.unravel_index(np.argmax(i_target[idx], axis=None), i_target[idx].shape)
                                 for idx in range(i_target.shape[0])] for i_target in target.numpy()]
            pred_landmarks = [[np.unravel_index(np.argmax(i_output[idx], axis=None), i_output[idx].shape)
                               for idx in range(i_output.shape[0])] for i_output in output[-1].numpy()] # only last one relevant for final prediction

            for idx, image in enumerate(data.numpy()):
                channels, height, width = image.shape

                img = cv2.resize(image[0], (width // 2, height // 2), cv2.INTER_CUBIC)

                image_target = np.copy(img)
                image_pred = np.copy(img)

                for (y, x) in target_landmarks[idx]:
                    image_target[y, x] = 1

                for (y, x) in pred_landmarks[idx]:
                    image_pred[y, x] = 1

                # do breakpoints here! first iteration is validation image, second is training
                Image.fromarray(image_target*255).show(f'{idx}_target')
                Image.fromarray(image_pred*255).show(f'{idx}_pred')

            #
            # end custom
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # args = parser.parse_args()
    config = ConfigParser(*parse_cmd_args(args))
    main(config, config.resume)
