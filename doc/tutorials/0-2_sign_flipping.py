"""
M/EEG: Sign Flipping
====================

Source reconstruction leaves the sign of each parcel (channel) time course
arbitrary: the same parcel can have opposite polarity in different sessions.
This is an identifiability problem in the source reconstruction of M/EEG that
cannot be avoided.

This matters when we pool sessions to train a model. The HMM and DyNeMo model
dynamic changes in the *covariance* of the data — and the sign of the
off-diagonal elements of the covariance depends on the (arbitrary) parcel
signs. If the signs are not aligned across sessions, the model sees spurious
covariance differences that are due to nothing more than a flipped sign. See
the :doc:`FAQ <../faq>` for more.

**Sign flipping** removes this ambiguity: for each session we search for the
per-parcel ``+1``/``-1`` vector whose (time-delay embedded) covariance best
matches a common **template**, and flip the parcels accordingly. This tutorial
covers:

1. Simulating parcellated data.
2. Visualizing the sign ambiguity across sessions.
3. What the template is.
4. Sign flipping with the default template: the median session.
5. Specifying a template, either from the same dataset or from another
   dataset.
6. Sign flipping arrays and fif files inside a processing pipeline.

Note, sign flipping is a *cross-session* step — it aligns sessions to each
other — so it is applied to a group of parcellated sessions after
preprocessing, source reconstruction and parcellation (see the
:doc:`MEG Processing tutorial <0-1_meg_preprocessing>`).
"""

#%%
# Simulating parcellated data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Rather than download a dataset, we simulate parcellated data for 5 sessions
# of 38 parcels. Each session is a mix of the *same* underlying spatial sources
# — so the sessions share a common covariance structure — but we give each
# session its own random per-parcel signs. This reproduces the situation in
# parcellated data: the covariance structure is shared across sessions, but the
# sign of each parcel is arbitrary and differs from session to session.
#
# We fix the random seed so the example is reproducible.

import numpy as np
from osl_dynamics.data import Data

rng = np.random.default_rng(42)

n_sessions = 5
n_samples = 8000
n_parcels = 38

# Shared spatial mixing, so every session has the same covariance structure
mixing = rng.standard_normal([n_parcels, n_parcels])

arrays = []
for _ in range(n_sessions):
    sources = rng.standard_normal([n_samples, n_parcels])
    parcels = sources @ mixing.T
    parcels *= rng.choice([-1, 1], size=n_parcels)  # arbitrary per-parcel signs
    arrays.append(parcels.astype(np.float32))

data = Data(arrays)
print(data)

#%%
# Visualizing the sign ambiguity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The sign ambiguity shows up in the covariance. We compare sessions using the
# covariance of the *time-delay embedded* data (adding time-lagged copies of
# each channel), which is the representation the HMM and DyNeMo are trained on.
#
# :func:`osl_dynamics.data.sign_flipping.calc_cov` computes this covariance for
# a session, and :func:`osl_dynamics.data.sign_flipping.calc_corr` measures how
# similar two covariances are (the correlation of their off-diagonal entries).
# Passing ``mode="abs"`` compares the *absolute* values, which is insensitive
# to the parcel signs.

from osl_dynamics.data import sign_flipping

# Settings for the covariance used in the sign-flip search
n_embeddings = 15
standardize = True

def similarity_matrix(covs, mode="signed"):
    """Session-by-session similarity of the covariances."""
    n = len(covs)
    matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            matrix[i, j] = sign_flipping.calc_corr(covs[i], covs[j], n_embeddings, mode=mode)
    return matrix

# Covariance of the time-delay embedded data for each session
ts = data.time_series()  # list of (n_samples, n_channels) arrays
covs = [sign_flipping.calc_cov(x, n_embeddings, standardize) for x in ts]

signed = similarity_matrix(covs)
absolute = similarity_matrix(covs, mode="abs")

#%%
# Let's plot both similarity matrices side by side.

import matplotlib.pyplot as plt

def plot_similarity(matrices, titles):
    fig, axes = plt.subplots(1, len(matrices), figsize=(10, 4.5))
    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xlabel("Session")
        ax.set_ylabel("Session")
        ax.set_xticks(range(n_sessions))
        ax.set_yticks(range(n_sessions))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

plot_similarity(
    [signed, absolute],
    ["Signed (sign-sensitive)", "Absolute (sign-invariant)"],
)

#%%
# The **absolute** matrix (right) is high everywhere: the sessions share the
# same underlying covariance *structure*. The **signed** matrix (left) is
# mixed — off-diagonal session pairs are weakly or negatively correlated. That
# difference is exactly the sign ambiguity: the structure is shared, but the
# signs are not aligned. Sign flipping aligns them.
#
# The template
# ^^^^^^^^^^^^
# Because the sign of each parcel is arbitrary, there is no "correct" sign to
# recover. All we need is for every session to use the *same* sign
# convention. So we pick a reference covariance — the **template** — and flip
# the parcels of each session so that its covariance best matches the
# template. Sessions that all match the same template then match each other.
#
# The template only sets which sign convention we end up with, so any
# covariance with the same structure as our sessions will do. There are two
# options:
#
# - **The default: the median session.** If you do not pass a template, one of
#   the sessions being flipped is used. You do not need anything else, and
#   this is all most analyses need.
# - **A template you specify.** You pass the template covariance yourself. It
#   can come from the *same* dataset (e.g. a reference session) or from
#   *another* dataset (e.g. the data used to train a pre-trained model).
#
# Default: the median session
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We do the sign flipping with the
# :meth:`Data.align_channel_signs <osl_dynamics.data.Data.align_channel_signs>`
# method. Without a template, it ranks the sessions by how similar their
# covariance is to all the other sessions (using the sign-invariant
# correlation) and uses the median session as the template. The log says which
# session was chosen. Every session is then flipped to match it, in place.
#
# We pass the same embedding settings as above, so the flips are found on the
# covariances we just plotted.

data.align_channel_signs(n_embeddings=n_embeddings, standardize=standardize)

#%%
# To check the effect, we recompute the signed similarity matrix from the
# (now flipped) data and compare it with the one before flipping.

covs_flipped = [
    sign_flipping.calc_cov(x, n_embeddings, standardize) for x in data.time_series()
]
signed_flipped = similarity_matrix(covs_flipped)

plot_similarity(
    [signed, signed_flipped],
    ["Signed, before flipping", "Signed, after flipping"],
)

#%%
# After flipping, every pair of sessions is positively correlated and the
# signed matrix looks like the absolute one: the signs are aligned and the
# sessions are ready to be pooled for training a model.
#
# Specifying a template
# ^^^^^^^^^^^^^^^^^^^^^
# The median session is picked from whichever sessions are in the Data
# object, so the sign convention can change if you rerun with sessions added
# or removed. Pass your own template when you need the sign convention to be
# fixed, e.g.
#
# - to add more sessions later and keep the signs of the sessions you have
#   already flipped,
# - to flip sessions one at a time in a processing pipeline (see
#   `Sign flipping arrays`_ and `MNE-Python FIF Files`_ below): there is no
#   batch of sessions to take a median from, so a template is required,
# - to match the signs of another dataset, e.g. the data used to train a
#   pre-trained model.
#
# The template is the covariance of the time-delay embedded data (see
# :func:`calc_cov <osl_dynamics.data.sign_flipping.calc_cov>`), built with the
# same ``n_embeddings`` and ``standardize`` settings used for the flipping.
# Pass it to ``align_channel_signs`` with ``template_cov`` (an array, or the
# path to a ``.npy`` file). Alternatively, pass a session's time series with
# ``template_data`` and the covariance is computed for you.
#
# A template from the same dataset
# ********************************
# The simplest choice is a session from the dataset you are flipping. Here we
# use session 0 as the reference and save its covariance, so we can reuse the
# same template every time we run the pipeline.
#
# :func:`sign_flipping.build_template
# <osl_dynamics.data.sign_flipping.build_template>` is a more robust
# alternative: it starts from the session most similar to all the others,
# aligns every session to it, and averages the aligned covariances. A template
# averaged over sessions is less noisy than any one session.
# :func:`sign_flipping.build_template_mne_raw
# <osl_dynamics.data.sign_flipping.build_template_mne_raw>` does the same from
# parcellated fif files.

# Start again from the original (unflipped) data
data = Data(arrays)

# Build the template from a reference session and save it
template_cov = sign_flipping.calc_cov(ts[0], n_embeddings, standardize)
np.save("template_cov.npy", template_cov)

# Correlation of each session with the template *before* flipping
before = np.array([sign_flipping.calc_corr(c, template_cov, n_embeddings) for c in covs])

#%%
# We pass the path to the saved template covariance to ``align_channel_signs``
# along with the same embedding settings we used to build it.

data.align_channel_signs(
    template_cov="template_cov.npy",
    n_embeddings=n_embeddings,
    standardize=standardize,
)

#%%
# Let's check the effect. We recompute each session's covariance from the
# (now flipped) data and measure its correlation with the template again.

ts_flipped = data.time_series()
covs_flipped = [sign_flipping.calc_cov(x, n_embeddings, standardize) for x in ts_flipped]
after = np.array([sign_flipping.calc_corr(c, template_cov, n_embeddings) for c in covs_flipped])

print("Correlation with template before:", np.round(before, 3))
print("Correlation with template after: ", np.round(after, 3))

#%%
# The reference session (session 0) is the template, so its correlation is 1
# and unchanged. The other sessions move up towards the template after
# flipping. Let's plot it.

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(n_sessions)
width = 0.4
ax.bar(x - width / 2, before, width, label="Before")
ax.bar(x + width / 2, after, width, label="After")
ax.set_xlabel("Session")
ax.set_ylabel("Correlation with template")
ax.set_xticks(x)
ax.legend()
fig.tight_layout()

#%%
# A template from another dataset
# *******************************
# Sometimes the template comes from a *different* dataset, e.g. the template
# covariance of a pre-trained model, which may have been recorded on another
# scanner, with different preprocessing, or a different population. You only
# need this if you want the signs to match that dataset: for an analysis of
# your own data, a template from the same dataset is enough.
#
# An individual session may match a template from another dataset only
# weakly, and the flip search becomes unreliable: one wrongly flipped parcel
# in one session is enough to add spurious covariance dynamics. Passing
# ``two_stage=True`` splits the problem in two:
#
# 1. Build a template from the sessions in the Data object
#    (:func:`sign_flipping.build_template
#    <osl_dynamics.data.sign_flipping.build_template>`, see above).
# 2. Find the *single* flip vector that aligns this dataset template to the
#    external template. Each session is then aligned to the dataset template
#    and this flip vector is applied on top.
#
# Each session only has to match its own dataset, which it does well, and the
# mismatch between the datasets is absorbed by one flip search on a clean,
# averaged covariance. The log reports the correlation of the dataset template
# with the external template, which is a useful summary of how well the new
# dataset matches the one the template came from.
#
# .. code-block:: python
#
#     data = Data(arrays)
#     data.align_channel_signs(
#         template_cov="other_dataset_template_cov.npy",
#         n_embeddings=15,
#         standardize=True,
#         two_stage=True,
#     )
#
# With a single session the two approaches are identical. Passing
# ``two_stage=True`` without a template aligns the sessions to the dataset
# template as it is, which is a more robust alternative to the median session.

#%%
# Sign flipping arrays
# ^^^^^^^^^^^^^^^^^^^^
# In a processing pipeline you often hold a single session's parcel data in
# memory as a ``(parcels, time)`` array — for example straight out of
# parcellation, before it is saved to a fif.
# :func:`sign_flipping.sign_flip <osl_dynamics.data.sign_flipping.sign_flip>`
# flips such an array directly against a template covariance. It returns the
# flipped data, the per-parcel ``+1``/``-1`` signs, and the correlation with
# the template.
#
# Here we flip one session against the template we saved earlier, transposing
# our simulated data to ``(parcels, time)``.

# Parcel data for one session, shape (parcels, time)
parcel_data = ts[1].T

flipped_data, flips, corr = sign_flipping.sign_flip(
    parcel_data,
    template_cov="template_cov.npy",
    n_embeddings=n_embeddings,
    standardize=standardize,
)

print("Correlation with template before flipping:", round(float(before[1]), 3))
print("Correlation with template after flipping: ", round(float(corr), 3))
print("Signs applied:", flips)

#%%
# The correlation jumps from near zero to close to the template, using the
# same search that :meth:`Data.align_channel_signs
# <osl_dynamics.data.Data.align_channel_signs>` runs for every session.
#
# MNE-Python FIF Files
# ^^^^^^^^^^^^^^^^^^^^
# In a batch processing pipeline you often want to sign flip each session's
# parcellated fif independently — for example inside
# ``osl_dynamics.meeg.parallel.run``.
# :func:`sign_flipping.sign_flip_mne_raw
# <osl_dynamics.data.sign_flipping.sign_flip_mne_raw>` wraps ``sign_flip``
# for this: pass a parcellated fif (a path or an ``mne.io.Raw``) and a
# template covariance (a path or an array), and it reads the parcel data,
# flips it, and saves the sign-flipped fif.
#
# Each file is flipped on its own, so there is no median session to fall back
# on and you must pass a template. It does not have to come from another
# dataset: build it from the parcellated fifs of the dataset you are flipping
# (or a subset of them) with :func:`sign_flipping.build_template_mne_raw
# <osl_dynamics.data.sign_flipping.build_template_mne_raw>` and save it.
#
# .. code-block:: python
#
#     from glob import glob
#     import numpy as np
#     from osl_dynamics.data import sign_flipping
#
#     parc_fifs = sorted(glob("derivatives/*/lcmv-parc-raw.fif"))
#
#     # Template from this dataset
#     template_cov, _ = sign_flipping.build_template_mne_raw(parc_fifs, n_jobs=8)
#     np.save("template_cov.npy", template_cov)
#
#     for parc_fif in parc_fifs:
#         sflip_parc_fif = parc_fif.replace("lcmv-parc-raw.fif", "sflip-lcmv-parc-raw.fif")
#         sign_flipping.sign_flip_mne_raw(
#             parc_fif,
#             template_cov="template_cov.npy",
#             output_file=sflip_parc_fif,
#         )
#
# If you omit the output file, the sign-flipped data is returned as an
# ``mne.io.Raw`` object instead of being written to disk::
#
#     raw = sign_flipping.sign_flip_mne_raw(parc_fif, "template_cov.npy")
#
# Only the parcel channels are flipped; every other channel, the annotations
# and the data in bad segments are kept (bad segments are left out of the
# covariance).
#
# To match a template from another dataset, use the two-stage alignment: find
# the flips that align the dataset template to the external template, and pass
# them as ``template_flips``:
#
# .. code-block:: python
#
#     template_flips, corr = sign_flipping.find_flips(
#         template_cov, np.load("other_dataset_template_cov.npy"), n_embeddings=15
#     )
#
#     for parc_fif in parc_fifs:
#         sign_flipping.sign_flip_mne_raw(
#             parc_fif,
#             template_cov=template_cov,
#             output_file=parc_fif.replace("lcmv-parc-raw.fif", "sflip-lcmv-parc-raw.fif"),
#             template_flips=template_flips,
#         )
#
# For EEG, pass ``frequency_bands=[(1, 4), (4, 8), (8, 13), (13, 30)]`` to
# ``build_template_mne_raw`` and ``sign_flip_mne_raw``: the flips are then
# found on the covariances of each band, pooled together, which carry more
# sign information than the broadband covariance. A template from another
# dataset must be stacked across the same bands.
#
# Next steps
# ^^^^^^^^^^
# The sign-flipped data can now be prepared and used to train a model:
#
# - :doc:`Preparing Data <../tutorials_build/1-2_data_prepare_meg>`.
# - :doc:`Training an HMM <../tutorials_build/3-2_hmm_training>`.
# - :doc:`Training DyNeMo <../tutorials_build/3-3_dynemo_training>`.
