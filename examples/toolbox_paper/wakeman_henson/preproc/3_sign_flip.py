"""Wakeman-Henson: Dipole Sign Flip.

"""

from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Directories
raw_dir = "data/ds117"
src_dir = "data/src"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel workers
    client = Client(n_workers=20, threads_per_worker=1)

    # Get subjects which we successfully source reconstructed
    subjects = []
    for path in sorted(glob(f"{src_dir}/*/rhino/parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    # Find a good template subject to match others to
    template = source_recon.find_template_subject(
        src_dir, subjects, n_embeddings=15, standardize=True
    )

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
    source_recon.run_src_batch(config, src_dir, subjects, dask_client=True)
