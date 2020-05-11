from taser.helpers import array_ops
from taser.inference import inference_functions
import numpy as np


def override_dict_defaults(default_dict: dict, override_dict: dict = None) -> dict:
    if override_dict is None:
        override_dict = {}

    return {**default_dict, **override_dict}


def process_data(dataset_parameters):
    raw_data = np.load(dataset_parameters["input_data"]).astype(np.float32)

    retrialed_data = raw_data[
        :, : dataset_parameters["trial_cutoff"], :: dataset_parameters["trial_skip"]
    ]
    concatenated_data = array_ops.trials_to_continuous(retrialed_data)

    events = concatenated_data[dataset_parameters["event_channel"]]
    input_data = concatenated_data[dataset_parameters["data_start"] :]

    if dataset_parameters["standardize"]:
        input_data = inference_functions.scale(input_data)
    if dataset_parameters["pca"]:
        input_data = inference_functions.pca(
            input_data, n_components=dataset_parameters["n_pcs"]
        )
    if dataset_parameters["standardize_pcs"]:
        input_data = inference_functions.scale(input_data)

    return input_data, events
