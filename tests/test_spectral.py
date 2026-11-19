"""Compare the spectral analysis functions against saved reference outputs."""

from pathlib import Path

import numpy as np
import pytest

from spectral_cases import cases

REFERENCE_FILE = Path(__file__).parent / "data" / "spectral_references.npz"
CASES = cases()


@pytest.fixture(scope="module")
def references():
    with np.load(REFERENCE_FILE) as f:
        return dict(f)


def _reference_outputs(references, name):
    outputs = []
    while f"{name}/{len(outputs)}" in references:
        outputs.append(references[f"{name}/{len(outputs)}"])
    return outputs


@pytest.mark.parametrize("name", list(CASES))
def test_matches_reference(name, references):
    expected = _reference_outputs(references, name)
    assert expected, f"no reference outputs for {name}"

    actual = CASES[name]()
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a.shape == e.shape
        assert np.iscomplexobj(a) == np.iscomplexobj(e)
        scale = np.nanmax(np.abs(e)) if np.any(np.isfinite(e)) else 1
        np.testing.assert_allclose(a, e, rtol=1e-4, atol=1e-6 * scale)
