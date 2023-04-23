"""Nottingham MRC MEGUK: Sign flipping.

"""

from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Directories
SRC_DIR = "data/src"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel workers
    client = Client(n_workers=16, threads_per_worker=1)

    # Subjects to sign flip
    subjects = []
    for path in sorted(glob(SRC_DIR + "/*/rhino/parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    # Find a good template subject to align other subjects to
    template = source_recon.find_template_subject(
        SRC_DIR, subjects, n_embeddings=15, standardize=True
    )

    # Settings
    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 3
            n_iter: 3000
            max_flips: 20
    """

    # Run batch sign flipping
    source_recon.run_src_batch(config, SRC_DIR, subjects, dask_client=True)
