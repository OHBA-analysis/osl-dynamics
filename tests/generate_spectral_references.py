"""Save the reference outputs used by test_spectral.py.

Usage::

    python tests/generate_spectral_references.py
"""

from pathlib import Path

import numpy as np

from spectral_cases import cases

REFERENCE_FILE = Path(__file__).parent / "data" / "spectral_references.npz"


def main():
    references = {}
    for name, fn in cases().items():
        print(name)
        for i, output in enumerate(fn()):
            references[f"{name}/{i}"] = output
    REFERENCE_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE_FILE, **references)
    print(f"Saved {len(references)} arrays to {REFERENCE_FILE}")


if __name__ == "__main__":
    main()
