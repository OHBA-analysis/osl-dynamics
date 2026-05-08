"""Nottingham MEGUK: Save Training Data for Dynamic Network Analysis."""

from glob import glob
from osl_dynamics.data import Data

outdir = "data/preproc"
files = sorted(glob(f"{outdir}/*/sflip-lcmv-parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit")
data.save("training_data/networks")
