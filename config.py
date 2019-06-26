CONFIG = {
    "name": "XRay",
    "n_gpu": 1,
    "arch": {
        "type": "ConvolutionalPoseMachines",
        "args": {
            "x_channels": 128,
            "stage_channels": 512,
            "num_stages": 5,
            "num_classes": 23,
            "depthwise_separable_convolution": False
        }
    },
    "data_loader": {
        "type": "XRayDataLoader",
        "args": {
            "data_dir": "data/XRay/Patient_0",
            "batch_size": 1,
            "shuffle": False,
            "validation_split": 0.4,
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
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 20,
        "tensorboardX": True
    },
    'fraction_of_dataset': 0.1,
    'sigma': 80,
    'threshold': 0.01,
    'prediction_blur': 1,
    'rescale_X_input': 256,
    'rescale_Y_input': 256,
    'rescale_X_target': 32,
    'rescale_Y_target': 32,
}
