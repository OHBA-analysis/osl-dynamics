DEFAULT_CONFIGS = {
    "hmm": (
        {
            "sequence_length": 2000,
            "batch_size": 32,
            "learning_rate": 0.01,
            "n_epochs": 20,
        },
        {"n_init": 3, "n_epochs": 1},
    ),
    "dynemo": (
        {
            "sequence_length": 200,
            "inference_n_units": 64,
            "inference_normalization": "layer",
            "model_n_units": 64,
            "model_normalization": "layer",
            "learn_alpha_temperature": True,
            "initial_alpha_temperature": 1.0,
            "do_kl_annealing": True,
            "kl_annealing_curve": "tanh",
            "kl_annealing_sharpness": 10,
            "n_kl_annealing_epochs": 20,
            "batch_size": 128,
            "learning_rate": 0.01,
            "lr_decay": 0.1,
            "n_epochs": 40,
        },
        {"n_init": 5, "n_epochs": 2, "take": 1},
    ),
    "swc": (
        {
            "window_length": 100,
            "window_offset": 100,
            "window_type": "rectangular",
            "learn_means": False,
            "learn_covariances": True,
        },
        {},
    ),
}
