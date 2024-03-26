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
    batch_size=64,
    learning_rate=0.005,
    n_epochs=20,
    learn_trans_prob=True,
)

sim = simulation.MSess_HMM_MVN(
    n_samples=3000,
    trans_prob="sequence",
    session_means="zero",
    session_covariances="random",
    n_states=config.n_states,
    n_channels=config.n_channels,
    n_covariances_act=2,
    n_sessions=100,
    embeddings_dim=2,
    spatial_embeddings_dim=2,
    embeddings_scale=0.001,
    n_groups=3,
    between_group_scale=0.2,
    stay_prob=0.9,
)
sim.standardize()

# Plot the embeddings
sim_se = sim.embeddings
assigned_groups = sim.assigned_groups
group_masks = [assigned_groups == i for i in range(sim.n_groups)]
plotting.plot_scatter(
    [sim_se[group_mask, 0] for group_mask in group_masks],
    [sim_se[group_mask, 1] for group_mask in group_masks],
    x_label="dim_1",
    y_label="dim_2",
    annotate=[
        np.array([str(i) for i in range(sim.n_sessions)])[group_mask]
        for group_mask in group_masks
    ],
    filename="figures/sim_embeddings.png",
)
training_data = data.Data([tc for tc in sim.time_series])

model = Model(config)
model.random_state_time_course_initialization(training_data, n_epochs=2, n_init=3)
model.fit(training_data)
alpha = model.get_alpha(training_data)
argmax_alpha = inference.modes.argmax_time_courses(alpha)

# Get the Fisher kernel matrix
fk = fisher_kernel.FisherKernel(model)
kernel = fk.get_kernel_matrix(training_data)

# 5-fold cross validation
scores = []
kf = KFold(5, shuffle=True, random_state=234)
for train_index, validation_index in kf.split(range(sim.n_sessions)):
    kernel_train = kernel[train_index][:, train_index]
    labels_train = assigned_groups[train_index]
    kernel_validation = kernel[validation_index][:, train_index]
    labels_validation = assigned_groups[validation_index]

    clf = OneVsOneClassifier(SVC(kernel="precomputed"))
    clf.fit(kernel_train, labels_train)

    labels_predict = clf.predict(kernel_validation)
    scores.append(clf.score(kernel_validation, labels_validation))


print("Mean score:", np.mean(scores))
print("Score std:", np.std(scores))

training_data.delete_dir()
