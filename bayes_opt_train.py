from bayes_opt import BayesianOptimization

from config import CONFIG
from parse_config import ConfigParser
from train import main


def bayesian_opt(num_channels, num_stacks, num_blocks, kernel_size, sigma=5, prediction_blur=5, threshold=0.01,
                 epochs=20):
    CONFIG['arch']['args']['num_channels'] = 2 ** int(round(num_channels))
    CONFIG['arch']['args']['num_stacks'] = int(round(num_stacks))
    CONFIG['arch']['args']['num_blocks'] = int(round(num_blocks))
    CONFIG['arch']['args']['kernel_size'] = int(round(kernel_size)) * 2 + 1
    CONFIG['sigma'] = sigma
    CONFIG['prediction_blur'] = prediction_blur
    CONFIG['threshold'] = threshold
    CONFIG['trainer']['epochs'] = int(epochs)
    return - main(ConfigParser(CONFIG))


def run_bayes_opt(pbounds, init_points=10, n_iter=10):
    optimizer = BayesianOptimization(
        f=bayesian_opt,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)


ranges = {
    'num_channels': (6, 8),  # {64, 128, 256}
    'num_stacks': (2, 8),
    'num_blocks': (1, 7),
    'kernel_size': (1, 4),  # {3, 5, 7, 9}
    # 'sigma': (1, 10),
    # 'prediction_blur': (1, 10),
    # 'threshold': (0.00001, 0.2)
    'epochs': (1, 3)
}
run_bayes_opt({
    'num_channels': (6, 8),  # {64, 128, 256}
    'num_stacks': (2, 7),
    'num_blocks': (1, 7),
    'kernel_size': (1, 4),  # {3, 5, 7, 9}
    'sigma': (0.6, 5),
    'prediction_blur': (0.01, 1),
    'threshold': (0.00001, 0.2),
    'epochs': (200, 200)

}, init_points=10, n_iter=10)