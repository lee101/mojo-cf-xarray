import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-cf-xarray.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-cf-xarray.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260926)


@pytest.fixture
def vertical_terms(rng):
    """Per-column terms for the ocean and atmosphere decoders.

    Every term is a full 1-D column of the same length, which is the shape
    `decode_vertical_coords` actually feeds the formulas when a coordinate
    varies with level.
    """
    n = 2000
    s = np.linspace(-1.0, 0.0, n)
    return {
        "s": s,
        "c": np.full(n, 0.5),
        "sigma": np.linspace(-1.0, 0.0, n),
        "lev": np.linspace(0.0, 12.0, n),
        "eta": np.full(n, 0.3),
        "depth": np.full(n, 4000.0),
        "zlev": np.linspace(0.0, -4000.0, n),
        "ps": np.full(n, 101325.0),
        "orog": rng.normal(size=n),
    }
