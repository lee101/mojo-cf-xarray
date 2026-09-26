"""Parity of the CF flag-variable mask extraction.

`_extract_flags` is the only bitwise operation the accessor layer performs
over user data. These tests build real `DataArray` flag variables and compare
the compiled kernel against the accessor's own output.
"""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import mojo_cf_xarray as mcf

BIT = 1


def flag_da(values, masks, meanings=None, name="flags"):
    """A CF flag variable; `flag_meanings` must be a space-joined string."""
    if meanings is None:
        meanings = [f"f{i}" for i in range(len(masks))]
    return xr.DataArray(
        np.asarray(values),
        dims=("x",),
        attrs={"flag_masks": list(masks), "flag_meanings": " ".join(meanings)},
        name=name,
    )


def test_extract_flags_matches_the_accessor(rng):
    meanings = ["none", "low", "high", "suspect"]
    values = rng.integers(0, 4, size=1000).astype(np.int32)
    da = flag_da(values, [1, 2, 4, 8], meanings)
    masks = mcf.flags.flag_dict(da)
    ours = mcf.extract_flags(values, masks)
    theirs = da.cf._extract_flags(meanings)
    for meaning in meanings:
        assert meaning in theirs
        assert_array_equal(ours[meaning], theirs[meaning].values)


def test_extract_flags_uses_the_equality_branch_when_values_are_given(rng):
    """A mask with a declared value tests `x & mask == value`, not truthiness."""
    meanings = ["one", "two", "four"]
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 4], dtype=np.int32)
    masks = {"one": (1, 1), "two": (2, 2), "four": (4, 4)}
    for meaning, (mask, value) in masks.items():
        expect = (values.astype(np.int64) & np.int64(mask)) == np.int64(value)
        assert_array_equal(mcf.extract_flags(values, masks)[meaning], expect)
        assert value is not None


def test_extract_flags_uses_the_truthiness_branch_for_valueless_masks(rng):
    """`flag_masks` entries with no `flag_values` set the flag on any bit."""
    da = flag_da(np.array([0, 1, 2, 3], dtype=np.int32), [1, 2], ["a", "b"])
    masks = {"a": (1, None), "b": (2, None)}
    got = mcf.extract_flags(np.array([0, 1, 2, 3], dtype=np.int32), masks)
    assert_array_equal(got["a"], [False, True, False, True])
    assert_array_equal(got["b"], [False, False, True, True])
    assert da is not None


def test_extract_flags_handles_a_mixed_set_of_masks(rng):
    meanings = ["clear", "cloud", "rain", "snow"]
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int16)
    da = flag_da(values, [1, 2, 4, 8], meanings)
    masks = mcf.flags.flag_dict(da)
    ours = mcf.extract_flags(values, masks)
    theirs = da.cf._extract_flags(meanings)
    for meaning in meanings:
        assert_array_equal(ours[meaning], theirs[meaning].values)


def test_extract_flags_is_exact_for_integer_input(rng):
    """The operation is a bitwise AND, so equality is the right tolerance."""
    for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16):
        values = rng.integers(0, 256, size=512).astype(dtype)
        masks = {"low": (1, 1), "mid": (14, 4), "high": (112, 32), "any": (255, None)}
        got = mcf.extract_flags(values, masks)
        x = values.astype(np.int64)
        assert_array_equal(got["low"], (x & 1) == 1)
        assert_array_equal(got["mid"], (x & 14) == 4)
        assert_array_equal(got["high"], (x & 112) == 32)
        assert_array_equal(got["any"], (x & 255) != 0)


def test_extract_flags_rejects_a_float_variable():
    with pytest.raises(TypeError):
        mcf.extract_flags(np.array([1.0, 2.0]), {"a": (1, 1)})


def test_extract_flags_with_no_flags_returns_empty():
    assert mcf.extract_flags(np.array([1, 2, 3], dtype=np.int32), {}) == {}


def test_a_zero_mask_never_fires():
    """`flag_masks = 0` is not the same as a missing mask.

    Upstream only takes the plain-equality branch when the mask attribute is
    absent, so a zero mask still goes through the bitwise path and
    `x & 0 == value` is false for every element. Pinning it stops a
    falsy-check from quietly becoming an equality test.
    """
    values = np.array([0, 1, 2, 3], dtype=np.int32)
    da = flag_da(values, [0], ["a"])
    da.attrs["flag_values"] = [2]
    masks = mcf.flags.flag_dict(da)
    got = mcf.extract_flags(values, masks)
    assert_array_equal(got["a"], da.cf._extract_flags(["a"])["a"].values)
    assert not got["a"].any()


def test_extract_flags_preserves_shape():
    values = np.arange(12, dtype=np.int32).reshape(3, 4)
    masks = {"even": (1, 0), "odd": (1, None)}
    got = mcf.extract_flags(values, masks)
    assert got["even"].shape == (3, 4)
    assert_array_equal(got["odd"], (values.astype(np.int64) & 1) != 0)
