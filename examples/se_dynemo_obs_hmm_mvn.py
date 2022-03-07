"""Example script for fitting a subject embedding observation model to data."""

print("Setting up")
from sklearn.decomposition import PCA
from ohba_models import data, files, simulation
from ohba_models.inference import tf_ops
from ohba_models.models.se_dynemo_obs import Config, Model
from ohba_models.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Hyperparameters
config = Config(
    n_modes=5,
    n_channels=40,
    n_subjects=10,
    embedding_dim=3,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=50,
)

# Simulate data
print("Simulating data")
sim = simulation.MultiSubject_HMM_MVN(
    n_samples=12800,
    n_subjects=config.n_subjects,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
    random_seed=123,
)
meg_data = data.Data([ts for ts in sim.time_series])

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    alpha=[mtc for mtc in sim.mode_time_course],
    subj_id=True,
    shuffle=True,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=config.n_epochs)

# Get the subject embeddings
subject_embeddings = model.get_subject_embeddings()

# Perform PCA on the subject embeddings to visualise the embeddings
pca = PCA(n_components=2)
pca.fit_transform(subject_embeddings)
print("explained variances ratio:", pca.explained_variance_ratio_)
plotting.plot_scatter(
    [subject_embeddings[:, 0]],
    [subject_embeddings[:, 1]],
    x_label="PC1",
    y_label="PC2",
    filename="subject_embeddings.png",
)

# Inferred covariances
group_means, group_covariances = model.get_group_means_covariances()
plotting.plot_matrices(group_covariances - sim.covariances, filename="cov_diff.png")
