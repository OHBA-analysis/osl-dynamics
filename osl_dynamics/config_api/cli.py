"""Functions for running full pipelines via the config API."""

import argparse

from osl_dynamics.config_api.pipeline import run_pipeline_from_file


def pipeline():
    """Run a pipeline from a config file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the config file.",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--restrict",
        type=str,
        help="GPU to use. Optional.",
    )
    args = parser.parse_args()

    run_pipeline_from_file(
        args.config_file,
        args.output_directory,
        restrict=args.restrict,
    )
