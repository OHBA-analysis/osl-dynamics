from osl_dynamics import simulation, data, inference
from osl_dynamics.analysis import fisher_kernel
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import KFold

os.makedirs("figures", exist_ok=True)

inference.tf_ops.gpu_growth()

config = Config(
    n_states=5,
    n_channels=20,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.005,
    n_epochs=20,
    learn_trans_prob=True,
)

sim = simulation.MSubj_HMM_MVN(
    n_samples=3000,
    trans_prob="sequence",
    subject_means="zero",
    subject_covariances="random",
    n_states=config.n_states,
    n_channels=config.n_channels,
    n_covariances_act=2,
    n_subjects=100,
    n_subject_embedding_dim=2,
    n_mode_embedding_dim=2,
    subject_embedding_scale=0.001,
    n_groups=3,
    between_group_scale=0.2,
    stay_prob=0.9,
    random_seed=1234,
)
sim.standardize()

# Plot the subject embeddings
sim_se = sim.subject_embeddings
assigned_groups = sim.assigned_groups
group_masks = [assigned_groups == i for i in range(sim.n_groups)]
plotting.plot_scatter(
    [sim_se[group_mask, 0] for group_mask in group_masks],
    [sim_se[group_mask, 1] for group_mask in group_masks],
    x_label="dim_1",
    y_label="dim_2",
    annotate=[
        np.array([str(i) for i in range(sim.n_subjects)])[group_mask]
        for group_mask in group_masks
    ],
    filename="figures/sim_subject_embeddings.png",
)
training_data = data.Data([tc for tc in sim.time_series])

model = Model(config)
model.random_state_time_course_initialization(training_data, n_epochs=2, n_init=3)
model.fit(training_data)
alpha = model.get_alpha(training_data)
argmax_alpha = inference.modes.argmax_time_courses(alpha)

# Dual estimation
time_series = training_data.time_series()
means = np.zeros([sim.n_subjects, config.n_states, config.n_channels])
covariances = np.array([[np.eye(config.n_channels)] * config.n_states] * sim.n_subjects)
for n in range(sim.n_subjects):
    masked_ts = np.expand_dims(time_series[n], 1) * np.expand_dims(alpha[n], -1)
    masked_ts = np.transpose(masked_ts, (1, 0, 2))
    for i, ts in enumerate(masked_ts):
        state_ts = ts[~np.all(np.isclose(ts, 0), axis=1)]
        if config.learn_means:
            means[n, i] = np.mean(state_ts, axis=0)
        if config.learn_covariances:
            covariances[n, i] = np.cov(state_ts, rowvar=False)

fk = fisher_kernel.FisherKernel(model)
kernel = fk.get_kernel_matrix(training_data, means, covariances)
# 5-fold cross validation
scores = []
kf = KFold(5, shuffle=True, random_state=234)
for train_index, validation_index in kf.split(range(sim.n_subjects)):
    kernel_train = kernel[train_index][:, train_index]
    labels_train = assigned_groups[train_index]
    kernel_validation = kernel[validation_index][:, train_index]
    labels_validation = assigned_groups[validation_index]

    clf = OneVsOneClassifier(SVC(kernel="precomputed"))
    clf.fit(kernel_train, labels_train)

    labels_predict = clf.predict(kernel_validation)
    wrong_subjects = validation_index[labels_predict != labels_validation]
    scores.append(clf.score(kernel_validation, labels_validation))
    print(wrong_subjects)

print("Mean score:", np.mean(scores))
print("Score std:", np.std(scores))

training_data.delete_dir()
