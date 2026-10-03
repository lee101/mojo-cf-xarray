"""The FFI contract: arity, contiguity and 64-bit addresses.

These are the failure modes that pass every small test and then break in
production, so they get their own file.
"""

import ctypes

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mojo_cf_xarray as mcf
from mojo_cf_xarray._lib import _SIGNATURES, addr, lib


def test_addr_rejects_non_contiguous_input():
    a = np.arange(24, dtype=np.float64).reshape(4, 6)
    assert addr(a) > 0
    with pytest.raises(ValueError):
        addr(a[:, ::2])


def test_every_symbol_checks_its_arity():
    """ctypes zero-fills a short argtypes list, so a dropped argument truncates."""
    with pytest.raises(TypeError):
        lib.guess_bounds_1d(4, 0)
    with pytest.raises(TypeError):
        lib.get_core_dim_order(4, 0, 0, 0)
    with pytest.raises(TypeError):
        lib.extract_flags(4, 1, 0, 0, 0, 0)


def test_buffers_above_four_gib_round_trip():
    """A `c_int` slot in argtypes would truncate the address and fault here."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    n = 8192
    size = n * 8
    want = 0x7F00_0000_0000
    addr_here = libc.mmap(None, size, 3, 0x02 | 0x20, -1, 0)
    assert addr_here
    if int(addr_here) <= 2**32:
        libc.munmap(addr_here, size)
        addr_here = libc.mmap(
            ctypes.c_void_p(want), size, 3, 0x02 | 0x20 | 0x10, -1, 0
        )
        assert addr_here == want, "could not map above 4 GiB"
    assert int(addr_here) > 2**32
    try:
        mapping = (ctypes.c_double * n).from_address(int(addr_here))
        for i in range(n):
            mapping[i] = float(i)
        coord = np.ascontiguousarray(np.asarray(mapping))
        assert coord.ctypes.data == int(addr_here)
        assert coord[-1] == float(n - 1)
        bounds = mcf.guess_bounds_1d(coord)
        assert_allclose(bounds[:, 0], coord - 0.5, rtol=0, atol=0)
        assert_allclose(bounds[:, 1], coord + 0.5, rtol=0, atol=0)
        assert mcf.axis_order(coord) == "ascending"
        assert mcf.is_bounds_monotonic(bounds)
    finally:
        libc.munmap(addr_here, size)


def test_every_export_is_declared_and_arity_checked():
    """A kernel that is not in the table silently has no argtypes at all."""
    expected = {
        "atmosphere_ln_pressure_decode",
        "atmosphere_sigma_decode",
        "atmosphere_hybrid_sigma_pressure_decode",
        "atmosphere_hybrid_height_decode",
        "atmosphere_sleve_decode",
        "ocean_sigma_decode",
        "ocean_s_decode",
        "ocean_sg1_decode",
        "ocean_sg2_decode",
        "ocean_sigma_z_decode",
        "ocean_double_sigma_decode",
        "guess_bounds_1d",
        "guess_bounds_2d",
        "get_core_dim_order",
        "extract_flags",
        "simd_width",
        "max_workers",
        "bounds_helper_2d",
        "ordered_vertices",
        "sort_vertices",
        "vertices_to_bounds_1d",
        "vertices_to_bounds_2d",
    }
    assert set(_SIGNATURES) == expected
    for name, (_restype, argtypes) in _SIGNATURES.items():
        assert getattr(lib, name).arity == len(argtypes)
