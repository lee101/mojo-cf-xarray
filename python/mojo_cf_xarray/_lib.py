"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults once a buffer lands above 4 GiB.

Each symbol is wrapped in a `_Checked` callable that refuses an argument count
other than the declared one: ctypes zero-fills a short `argtypes` list, so a
dropped argument would otherwise be truncated silently.
"""

from __future__ import annotations

import ctypes
import pathlib
from types import SimpleNamespace

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-cf-xarray.so"

_I64 = ctypes.c_int64
_F64 = ctypes.c_double

_SIGNATURES: dict[str, tuple[object, list[object]]] = {
    "cfx_affine_decode": (None, [_I64, _I64, _I64, _I64, _I64, _I64]),
    "cfx_ln_pressure_decode": (None, [_I64, _I64, _I64, _I64]),
    "cfx_hybrid_sigma_decode": (None, [_I64] * 5 + [_F64] + [_I64] * 2 + [_F64] + [_I64] * 2),
    "cfx_sleve_decode": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _I64, _I64]),
    "cfx_s_coordinate_decode": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _F64, _I64, _I64, _F64, _I64, _I64, _F64, _I64]),
    "cfx_sg1_decode": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _I64, _F64, _I64]),
    "cfx_sg2_decode": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _I64, _F64, _I64]),
    "cfx_sigma_z_decode": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _F64, _I64, _I64]),
    "cfx_double_sigma_decode": (None, [_I64, _I64, _I64, _I64, _I64, _F64, _I64, _I64, _F64, _I64, _I64, _F64, _I64, _I64, _F64, _I64, _I64, _F64, _I64]),
    "cfx_guess_bounds_1d": (None, [_I64, _I64, _I64]),
    "cfx_axis_order": (None, [_I64, _I64, _I64]),
    "cfx_extract_flags": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _I64]),
}


class _Checked:
    """A ctypes function that verifies its call arity before dispatching."""

    __slots__ = ("_fn", "_name", "_n")

    def __init__(self, fn, name: str, n: int):
        self._fn = fn
        self._name = name
        self._n = n

    @property
    def arity(self) -> int:
        return self._n

    def __call__(self, *args):
        if len(args) != self._n:
            raise TypeError(
                f"{self._name} takes exactly {self._n} arguments, got {len(args)}"
            )
        return self._fn(*args)


def _load() -> SimpleNamespace:
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    cdll = ctypes.CDLL(str(_LIB_PATH))
    ns = SimpleNamespace()
    for name, (restype, argtypes) in _SIGNATURES.items():
        fn = getattr(cdll, name, None)
        if fn is None:
            raise RuntimeError(f"symbol {name} missing from {_LIB_PATH}")
        if len(argtypes) == 0:
            raise RuntimeError(f"empty argtypes for {name}")
        fn.restype = restype
        fn.argtypes = argtypes
        setattr(ns, name, _Checked(fn, name, len(argtypes)))
    return ns


lib = _load()


def addr(a: np.ndarray) -> int:
    """Address of a C-contiguous buffer, as the 64-bit integer the ABI wants."""
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError("buffer must be C-contiguous")
    return int(a.ctypes.data)


def f64(x) -> ctypes.c_double:
    return ctypes.c_double(float(x))
