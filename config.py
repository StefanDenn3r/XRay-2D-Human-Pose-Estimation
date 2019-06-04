CONFIG = {
    "name": "XRay",
    "n_gpu": 1,
    "arch": {
        "type": "StackedHourglassNet",
        "args": {
            "num_channels": 128,
            "num_stacks": 2,
            "num_blocks": 4,
            "kernel_size": 7
        }
    },
    "data_loader": {
        "type": "XRayDataLoader",
        "args": {
            "data_dir": "data/XRay/Ex3",
            "batch_size": 1,
            "shuffle": False,
            "validation_split": 0.5,
            "num_workers": 0
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.001,
            "weight_decay": 0,
            "amsgrad": True
        }
    },
    "loss": "smooth_l1_loss",
    "metrics": [
        "percentage_correct_keypoints",
        "keypoint_distance_loss",
        "mse_loss"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 200,
        "save_dir": "saved/",
        "save_period": 20,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 20,
        "tensorboardX": True
    },
    'fraction_of_dataset': 0.1,
    'sigma': 20,
    'threshold': 0.01,
    'prediction_blur': 1,
    'rescale_X': 256,
    'rescale_Y': 256
}
