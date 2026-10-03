"""Dataset-level entry points for the parametric vertical coordinate decode.

Upstream reaches both of these through the `.cf` accessor:
`Dataset.cf.formula_terms` and `Dataset.cf.decode_vertical_coords`. Neither
contains arithmetic: one walks the variables' `formula_terms` attribute, the
other dispatches to `cf_xarray.parametric.TRANSFORM_FROM_STDNAME`, which is
ported -- and compiled -- in `parametric.py` here. The names, argument order,
defaults, messages, attribute lookups and control flow below are upstream's,
including the messages and the misspelling in the `NotImplementedError`.

Where upstream scans the variables it resolves as the dataset's `Z`
dimensions, this scans every variable carrying a `formula_terms` attribute,
which is the same set for a CF-compliant file and keeps the
`standard_name`-less coordinate visible to the `continue` in
`decode_vertical_coords`. Term names are reported as written; upstream
lowercases them only when it builds the `terms` mapping for the decoder.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Hashable
from typing import TYPE_CHECKING

from . import parametric
from .parametric import OCEAN_STDNAME_MAP

if TYPE_CHECKING:
    from xarray import DataArray, Dataset

__all__ = ["decode_vertical_coords", "formula_terms"]

_SEPARATOR = re.compile(r"\s*:\s*")


def _parse_formula_terms(text: str) -> dict[str, str]:
    """`"a: s_rho b: s_rho"` -> `{"a": "s_rho", "b": "s_rho"}`."""
    terms = {}
    for mapping in _SEPARATOR.sub(":", text).split():
        key, value = mapping.split(":")
        terms[key] = value
    return terms


def formula_terms(ds: Dataset) -> dict[Hashable, dict[str, str]]:
    """Mapping of parametric coordinate name to its standard term names.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to scan.

    Returns
    -------
    dict
        Dictionary of the form
        ``{parametric_coord_name: {standard_term_name: variable_name}}``.

    Notes
    -----
    Upstream drops the terms whose variable is absent from `ds`; they are kept
    here so that `decode_vertical_coords` reports the absence.
    """
    results: dict[Hashable, dict[str, str]] = {}
    for name, var in ds._variables.items():
        text = var.attrs.get("formula_terms")
        if text is None:
            text = var.encoding.get("formula_terms")
        if not text:
            continue
        terms = _parse_formula_terms(text)
        if terms:
            results[name] = terms
    return results


def decode_vertical_coords(
    ds: Dataset,
    *,
    outnames: dict[str, str] | None = None,
    prefix: str | None = None,
) -> Dataset:
    """Decode parameterized vertical coordinates in place.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to decode.
    outnames : dict, optional
        Keys are the input sigma/s coordinate variable names, the values the
        names to use for the associated vertical coordinates.
    prefix : str, optional
        Prefix for newly created z variables, e.g. ``s_rho`` becomes
        ``z_rho``.

    Returns
    -------
    xr.Dataset
        `ds`, with a new coordinate per decoded parametric coordinate. The
        accessor version returns `None`; returning the dataset makes this
        usable without the accessor.

    See Also
    --------
    formula_terms

    Notes
    -----
    Will only decode when the ``formula_terms`` and ``standard_name``
    attributes are set on the parameter (e.g ``s_rho``).

    Two deliberate divergences from the accessor version, both consequences of
    scanning every variable rather than the dataset's `Z` dimensions. Both are
    tested in ``tests/test_accessor.py``.

    - The decoded coordinate inherits its source's ``formula_terms``
      attribute, because `_wrap` merges the operands' attributes the way an
      xarray binary op does -- upstream's decoded coordinate has it too. It
      names the same variables, and this scan reads every variable that
      carries one, so a repeat call also decodes the coordinate the previous
      call produced. Upstream never sees it as a `Z` dimension and never looks
      at it again. A repeat call here therefore needs an ``outnames`` key for
      every decoded coordinate as well as for every input, where upstream needs
      only the inputs.
    - A ``formula_terms`` naming a variable absent from `ds` is kept by
      ``formula_terms`` and reported by the ``KeyError`` below, where upstream
      drops it silently and decodes the coordinate from the terms it did keep.
    """
    for dim, terms_dict in formula_terms(ds).items():
        if prefix is None:
            assert outnames is not None, (
                "if prefix is None, outnames must be provided"
            )
            try:
                zname = outnames[dim]
            except KeyError:
                raise KeyError(
                    "Your `outnames` need to include a key of `dim`."
                ) from None
        else:
            warnings.warn(
                "`prefix` is being deprecated; use `outnames` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            zname = f"{prefix}_" + "_".join(dim.split("_")[1:])

        if "standard_name" not in ds[dim].attrs:
            continue
        stdname = ds[dim].attrs["standard_name"]

        # map "standard" formula term names to actual variable names
        terms: dict[str, DataArray] = {}
        for key, value in terms_dict.items():
            if value not in ds:
                raise KeyError(
                    f"Variable {value!r} is required to decode coordinate for {dim!r}"
                    " but it is absent in the Dataset."
                )
            # keys should be case insensitive
            terms[key.lower()] = ds[value]

        try:
            transform = parametric.TRANSFORM_FROM_STDNAME[stdname]
        except KeyError:
            # The decoded coordinate is the only variable this scan ever sees
            # that upstream would not: it carries its source's `formula_terms`
            # and one of the eleven *output* standard names, which is not in
            # `TRANSFORM_FROM_STDNAME`. Upstream never reaches this branch for
            # one, because a decoded coordinate is not a `Z` dimension. The
            # message is this port's, and says which coordinate it was rather
            # than naming a standard name that was never missing; every other
            # unknown standard name still gets upstream's message verbatim.
            decoded = set(OCEAN_STDNAME_MAP) | {"air_pressure"}
            if dim in ds.coords and stdname in decoded:
                raise NotImplementedError(
                    f"{dim!r} is already a decoded vertical coordinate "
                    f"({stdname!r}); there is nothing left to decode."
                ) from None
            raise NotImplementedError(
                f"Coordinate function for {stdname!r} not implmented yet. Contributions welcome!"
            ) from None

        ds.coords[zname] = transform.from_terms(terms).decode()

    return ds
