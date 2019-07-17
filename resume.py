import glob
import os
from importlib.machinery import SourceFileLoader

from parse_config import ConfigParser
from train import main


def resume(run_dir=None, model_pth=None):
    base_saved_dir = "saved/models/XRay"

    for temp_run_dir in os.listdir(base_saved_dir)[::-1]:
        if run_dir is None:
            run_path = os.path.join(base_saved_dir, temp_run_dir)
        else:
            run_path = os.path.join(base_saved_dir, run_dir)

        if model_pth is None:
            model_path_list = glob.glob(f'{run_path}/checkpoint-epoch*.pth')
            if not model_path_list:
                continue
            model_path = model_path_list[-1]
            break
        else:
            model_path = os.path.join(run_path, model_pth)
            break

    config = SourceFileLoader("CONFIG", os.path.join(run_path, 'config.py')).load_module().CONFIG
    epoch = int(model_path.split('checkpoint-epoch')[-1][:-4])
    for i in epoch:
        config['data_loader']['args']['custom_args']['sigma_reduction_factor'] += config['data_loader']['args']['custom_args']['sigma_reduction_factor']* \
            config['data_loader']['args']['custom_args']['sigma_reduction_factor_change_rate']
        config['data_loader']['args']['custom_args']['sigma'] *= config['data_loader']['args']['custom_args']['sigma_reduction_factor'] 
    config['data_loader']['args']['custom_args']['sigma'] = max(20, config['data_loader']['args']['custom_args']['sigma'])
    main(ConfigParser(config, model_path))


resume()
