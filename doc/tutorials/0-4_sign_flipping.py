"""
Sign Flipping
=============

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
matches a common template, and flip the parcels accordingly. This tutorial
covers:

1. Simulating parcellated data.
2. Visualising the sign ambiguity across sessions.
3. Aligning the signs to a fixed template covariance.
4. Using the median session as a template instead.
5. The lower-level API used inside a processing pipeline.

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
# Visualising the sign ambiguity
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

# Covariance of the time-delay embedded data for each session
ts = data.time_series()  # list of (n_samples, n_channels) arrays
covs = [sign_flipping.calc_cov(x, n_embeddings, standardize) for x in ts]

# Session-by-session similarity of the covariances
n_sessions = len(covs)
signed = np.zeros([n_sessions, n_sessions])
absolute = np.zeros([n_sessions, n_sessions])
for i in range(n_sessions):
    for j in range(n_sessions):
        signed[i, j] = sign_flipping.calc_corr(covs[i], covs[j], n_embeddings)
        absolute[i, j] = sign_flipping.calc_corr(covs[i], covs[j], n_embeddings, mode="abs")

#%%
# Let's plot both similarity matrices side by side.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, matrix, title in zip(
    axes,
    [signed, absolute],
    ["Signed (sign-sensitive)", "Absolute (sign-invariant)"],
):
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(title)
    ax.set_xlabel("Session")
    ax.set_ylabel("Session")
    ax.set_xticks(range(n_sessions))
    ax.set_yticks(range(n_sessions))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()

#%%
# The **absolute** matrix (right) is high everywhere: the sessions share the
# same underlying covariance *structure*. The **signed** matrix (left) is
# mixed — off-diagonal session pairs are weakly or negatively correlated. That
# difference is exactly the sign ambiguity: the structure is shared, but the
# signs are not aligned. Sign flipping aligns them.
#
# Aligning to a template covariance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We align every session to a **fixed template covariance**. We build the
# template once from a chosen reference session and save it to disk.
#
# Using a fixed, saved template (rather than the median session of the current
# batch, which is the default) means the sign convention is identical every
# time you run the pipeline — so you can add more sessions later, or pool this
# dataset with another, and the signs still match. This reproducibility is why
# it is worth saving the template.

# Build the template from a reference session and save it
template_index = 0
template_cov = sign_flipping.calc_cov(ts[template_index], n_embeddings, standardize)
np.save("template_cov.npy", template_cov)

# Correlation of each session with the template *before* flipping
before = np.array([sign_flipping.calc_corr(c, template_cov, n_embeddings) for c in covs])

#%%
# We do the sign flipping with the
# :meth:`Data.align_channel_signs <osl_dynamics.data.Data.align_channel_signs>`
# method. We pass the path to the saved template covariance and the same
# embedding settings we used to build it. The method searches for the best
# per-parcel flips for each session and applies them in place.

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
# The sessions are now sign aligned and ready to be pooled for training a
# model.
#
# Using the median session as a template
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# If you do not pass a template, ``align_channel_signs`` picks the *median*
# session of the batch as the template automatically. This is convenient for a
# one-off analysis, but the choice of template then depends on which sessions
# happen to be in the batch — so it is not reproducible if you add or remove
# sessions later.
#
# .. code-block:: python
#
#     data = Data(arrays)
#     data.align_channel_signs(n_embeddings=15, standardize=True)
#
# MNE-Python FIF Files
# ^^^^^^^^^^^^^^^^^^^^
# In a batch processing pipeline you often want to sign flip each session's
# parcellated fif independently against a template you saved earlier
# — for example inside ``osl_dynamics.meeg.parallel.run``.
# :func:`sign_flipping.sign_flip_mne_raw
# <osl_dynamics.data.sign_flipping.sign_flip_mne_raw>` does this in one call:
# pass a parcellated fif (a path or an ``mne.io.Raw``) and a template
# covariance (a path or an array), and it saves the sign-flipped fif.
#
# .. code-block:: python
#
#     from glob import glob
#     from osl_dynamics.data import sign_flipping
#
#     for parc_fif in sorted(glob("derivatives/*/lcmv-parc-raw.fif")):
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
# Next steps
# ^^^^^^^^^^
# The sign-flipped data can now be prepared and used to train a model:
#
# - :doc:`Preparing Data <../tutorials_build/1-2_data_prepare_meg>`.
# - :doc:`Training an HMM <../tutorials_build/3-2_hmm_training>`.
# - :doc:`Training DyNeMo <../tutorials_build/3-3_dynemo_training>`.
