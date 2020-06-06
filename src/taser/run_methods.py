from typing import Union

import yaml

from taser import data_manipulation, plotting, tf_ops, trainers
from taser.inference.models import inference_rnn


def run_from_dict(settings: Union[dict, str]):
    if isinstance(settings, str):
        with open(settings, "r") as f:
            settings = yaml.load(f, Loader=yaml.Loader)

    data_settings = settings["data_parameters"]
    data_processing_settings = settings["data_processing"]
    dataset_settings = settings["dataset_parameters"]
    model_settings = settings["model_parameters"]
    trainer_settings = settings["trainer_parameters"]
    hyperparameters = settings["hyperparams"]

    print("Importing data.")
    meg_data = data_manipulation.MEGData(**data_settings)

    meg_data.make_continuous()
    meg_data.standardize(**data_processing_settings)

    train_dataset, predict_dataset = tf_ops.train_predict_dataset(
        meg_data, **dataset_settings
    )

    # print("Fit GMM")
    # gmm = mixture.GaussianMixture(n_components=model_settings["n_states"])
    # gmm.fit(meg_data)
    # means = gmm.means_
    # covariances = gmm.covariances_
    #
    # cholesky = find_cholesky_decompositions(covariances, means, learn_means=False)
    #
    # model_settings["mus_initial"] = means
    # model_settings["cholesky_djs_initial"] = cholesky

    print("Creating model.")
    model_settings["n_channels"] = meg_data.shape[1]
    model = inference_rnn.InferenceRNN(**model_settings)

    print("Creating trainer.")
    trainer = trainers.AnnealingTrainer(model=model, **trainer_settings)

    print("Starting training.")
    trainer.train(train_dataset, n_epochs=hyperparameters["n_epochs"])
    trainer.plot_loss()

    inferred_state_time_course = trainer.predict_latent_variable(predict_dataset)

    print("Plotting results.")
    plotting.highlight_states(inferred_state_time_course, n_time_points=20000)

    plotting.plot_state_lifetimes(inferred_state_time_course)

    print("Done.")

    return meg_data, trainer
