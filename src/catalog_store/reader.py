"""
Cone searches on an offline copy of an astronomical catalog.

A *store* is a directory with one sub-directory per catalog. Each catalog has
a ``catalog.json`` manifest and its rows in Parquet files partitioned by
declination zone. ``docs/store-format.md`` in the repository describes the
layout in full; `CatalogStore` is the only thing most users need.

This module is self-contained on purpose: it needs only numpy, pyarrow and
astropy, and can be copied into another project as a single file.

The code was extracted from ``stellarphot.catalog_backends``
(https://github.com/feder-observatory/stellarphot), which also has the writer.
"""

import copy
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from astropy import units as u
from astropy.coordinates import angular_separation
from astropy.table import MaskedColumn, Table
from astropy.utils.exceptions import AstropyUserWarning

__all__ = [
    "STORE_FORMAT",
    "STORE_FORMAT_VERSION",
    "CatalogNotFoundError",
    "CatalogStore",
    "CatalogStoreWarning",
    "CatalogUnavailableError",
]

STORE_FORMAT = "stellarphot-catalog-store"
STORE_FORMAT_VERSION = 1

# Columns that exist only to partition the data; never returned.
_INDEX_COLS = ("zone",)


class CatalogNotFoundError(LookupError):
    """The store has no catalog with the requested identifier or name."""


class CatalogUnavailableError(RuntimeError):
    """
    The store has the catalog but its data could not be read, e.g. because
    the disk is not mounted, a file is corrupt or a declination zone is gone.
    """


class CatalogStoreWarning(AstropyUserWarning):
    """Something about a store is off, but a query can still be answered."""


class CatalogStore:
    """
    A directory of catalogs stored as declination-zone Parquet files.

    Parameters
    ----------
    root : str or `pathlib.Path`
        Directory containing one sub-directory per catalog, each with a
        ``catalog.json`` manifest.

    Attributes
    ----------
    root : `pathlib.Path`
        The directory the store was opened on.

    Notes
    -----
    Opening a store reads only the manifests. A sub-directory whose manifest
    cannot be used (unreadable, not this format, or written in a later format
    version) is skipped with a `CatalogStoreWarning`, so one bad catalog does
    not hide the others. A missing or empty ``root`` also only warns; the
    store then simply has no catalogs.

    Examples
    --------
    >>> from astropy import units as u
    >>> from astropy.coordinates import SkyCoord
    >>> store = CatalogStore("/srv/catalogs")              # doctest: +SKIP
    >>> stars = store.cone_search(                         # doctest: +SKIP
    ...     "refcat2",
    ...     SkyCoord(135.55, 49.84, unit="deg"),
    ...     10 * u.arcmin,
    ...     column_filters={"rmag": "<=16"},
    ... )
    """

    def __init__(self, root):
        self.root = Path(root)
        # Manifest (plus a few derived entries) for each catalog, by identifier.
        self._catalogs = {}
        # Directory name -> identifier, so "refcat2" works as well as the id.
        self._ids_by_name = {}
        if not self.root.is_dir():
            warnings.warn(
                f"Catalog store {self.root} does not exist; "
                "no catalogs will be read from it.",
                CatalogStoreWarning,
                stacklevel=2,
            )
            return
        for manifest in sorted(self.root.glob("*/catalog.json")):
            try:
                info = _load_manifest(manifest)
            except (OSError, ValueError, KeyError, TypeError) as err:
                warnings.warn(
                    f"Skipping catalog with unusable manifest {manifest}: {err!r}",
                    CatalogStoreWarning,
                    stacklevel=2,
                )
                continue
            self._catalogs[info["vizier_id"]] = info
            self._ids_by_name[manifest.parent.name] = info["vizier_id"]
        if not self._catalogs:
            warnings.warn(
                f"Catalog store {self.root} contains no catalogs "
                "(no usable */catalog.json found).",
                CatalogStoreWarning,
                stacklevel=2,
            )

    def __repr__(self):
        return f"<CatalogStore {self.root} catalogs={list(self.catalog_ids)}>"

    def __contains__(self, catalog):
        return catalog in self._catalogs or catalog in self._ids_by_name

    @property
    def catalog_ids(self):
        """
        Identifiers of the catalogs in the store, sorted.

        Returns
        -------
        tuple of str
            The ``vizier_id`` of each catalog, e.g. ``"J/ApJ/867/105/refcat2"``.
        """
        return tuple(sorted(self._catalogs))

    def manifest(self, catalog):
        """
        Return a catalog's manifest.

        Parameters
        ----------
        catalog : str
            Catalog identifier, or the name of the catalog's directory.

        Returns
        -------
        dict
            A copy of the contents of the catalog's ``catalog.json``.

        Raises
        ------
        CatalogNotFoundError
            If the store has no such catalog.
        """
        info = self._info(catalog)
        return copy.deepcopy(
            {key: value for key, value in info.items() if not key.startswith("_")}
        )

    def is_complete(self, catalog, column_filters=None):
        """
        Report whether the store holds every source a query could return.

        Parameters
        ----------
        catalog : str
            Catalog identifier, or the name of the catalog's directory.
        column_filters : dict, optional
            The filters the query would use; see `cone_search`.

        Returns
        -------
        bool
            ``True`` if the store is as deep as the query, so that an empty
            answer means there are no such sources. ``False`` if the query may
            reach fainter than the store goes.

        Raises
        ------
        CatalogNotFoundError
            If the store has no such catalog.

        Notes
        -----
        A store built from only the bright part of a catalog records that in
        its manifest as ``mag_limit``, e.g. ``{"rmag": 19.0}``. A query is
        within such a store's depth only if one of its filters is an upper
        bound (``<`` or ``<=``) on a limited column, at or brighter than the
        limit.

        A range filter (``a..b``) is never counted, even if ``b`` is within
        the limit; that is conservative.

        With several limited columns one qualifying filter is enough. That
        matches a store cut on the brightest of the columns: a source fainter
        than the limit in one column is still present if it is brighter in
        another.
        """
        return _covers(self._info(catalog), column_filters or {})

    def cone_search(self, catalog, center, radius, column_filters=None):
        """
        Return the catalog rows within ``radius`` of ``center``.

        Parameters
        ----------
        catalog : str
            Catalog identifier, or the name of the catalog's directory.
        center : `astropy.coordinates.SkyCoord`
            Center of the cone: a single position, in any frame.
        radius : `astropy.units.Quantity` or float
            Angular radius of the cone. A bare number is taken as degrees.
        column_filters : dict, optional
            Limits on column values, keyed by column name, in the subset of
            VizieR's syntax given in the Notes, e.g. ``{"rmag": "<=16"}``. A
            row must pass every filter.

        Returns
        -------
        `astropy.table.Table`
            The matching rows, with the catalog's own column names and units.
            Every column is masked; a value the catalog leaves blank is
            masked. Rows are in no particular order. The table's ``meta`` has
            ``local_store`` (the data directory) and ``local_store_version``
            (the manifest's ``version``).

        Raises
        ------
        CatalogNotFoundError
            If the store has no such catalog.
        CatalogUnavailableError
            If the catalog's data cannot be read.
        ValueError
            If a filter is malformed or names a column the catalog lacks, if
            ``center`` is not a single position, or if ``radius`` is negative
            or not finite.

        Warns
        -----
        CatalogStoreWarning
            If the query may reach fainter than the store goes; see
            `is_complete`. The rows the store does have are still returned.

        Notes
        -----
        Filters are ``"<=x"``, ``"<x"``, ``">=x"``, ``">x"``, ``"=x"`` (or
        just ``"x"``), ``"!=x"`` and the inclusive range ``"a..b"``, with
        numeric ``x``. A blank (masked) value never passes a filter.

        Positions are compared as stored, with no proper-motion correction;
        the center is converted to ICRS first.
        """
        info = self._info(catalog)
        column_filters = dict(column_filters or {})
        if not center.isscalar:
            raise ValueError("center must be a single position")
        center = center.icrs
        ra0, dec0 = center.ra.deg, center.dec.deg
        radius = float(u.Quantity(radius, u.deg).to_value(u.deg))
        if not np.isfinite(radius) or radius < 0:
            raise ValueError(f"radius must be finite and not negative, got {radius}")

        ra_col, dec_col = info["ra_col"], info["dec_col"]
        zone_deg = info.get("zone_deg", 1.0)
        dec_min, dec_max = max(dec0 - radius, -90.0), min(dec0 + radius, 90.0)
        zone_lo = int(np.floor((dec_min + 90) / zone_deg))
        zone_hi = int(np.floor((dec_max + 90) / zone_deg))
        expr = (
            (pc.field("zone") >= zone_lo)
            & (pc.field("zone") <= zone_hi)
            & (pc.field(dec_col) >= dec_min)
            & (pc.field(dec_col) <= dec_max)
        )
        # An RA bounding box only makes sense away from the poles.
        if dec_max < 90 and dec_min > -90:
            half_width = radius / np.cos(np.radians(max(abs(dec_min), abs(dec_max))))
            if half_width < 180:
                lo, hi = ra0 - half_width, ra0 + half_width

                def ra_between(a, b):
                    return (pc.field(ra_col) >= a) & (pc.field(ra_col) <= b)

                if lo < 0:
                    expr &= ra_between(lo + 360, 360) | ra_between(0, hi)
                elif hi > 360:
                    expr &= ra_between(lo, 360) | ra_between(0, hi - 360)
                else:
                    expr &= ra_between(lo, hi)

        data_path = info["_path"]
        missing = _missing_zones(info, zone_lo, zone_hi)
        if missing:
            raise CatalogUnavailableError(
                f"Could not read {info['vizier_id']} at {data_path}: "
                f"zone(s) {', '.join(map(str, missing))} missing"
            )

        def unavailable(err):
            # Missing directory (disk unmounted), corrupt file, schema trouble.
            return CatalogUnavailableError(
                f"Could not read {info['vizier_id']} at {data_path}: {err}"
            )

        try:
            dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
        except Exception as err:
            raise unavailable(err) from err

        # A bad filter is a mistake in the query, not an unreadable store, so
        # it is raised as is rather than as CatalogUnavailableError.
        for col, spec in column_filters.items():
            if col not in dataset.schema.names:
                raise ValueError(f"Cannot filter on {col!r}: column not in catalog")
            col_expr = _filter_expression(
                pc.field(col), spec, dataset.schema.field(col).type
            )
            if col_expr is not None:
                expr &= col_expr

        try:
            arrow_table = dataset.to_table(filter=expr)
        except Exception as err:
            raise unavailable(err) from err

        # Exact cut on the sphere, before the (slower) conversion to astropy.
        sep = _angular_separation_deg(
            arrow_table[ra_col].to_numpy(zero_copy_only=False),
            arrow_table[dec_col].to_numpy(zero_copy_only=False),
            ra0,
            dec0,
        )
        arrow_table = arrow_table.filter(pa.array(sep <= radius))
        arrow_table = arrow_table.drop_columns(
            [c for c in arrow_table.column_names if c in _INDEX_COLS]
        )
        table = _arrow_to_masked_table(arrow_table, info.get("units") or {})
        table.meta["local_store"] = str(data_path)
        table.meta["local_store_version"] = info.get("version")

        # The scan already applied the filters; this stays the authority on
        # their semantics and is cheap on the rows that are left.
        table = _apply_column_filters(table, column_filters)

        if not _covers(info, column_filters):
            described = ", ".join(
                f"{col} <= {lim}" for col, lim in info["mag_limit"].items()
            )
            warnings.warn(
                f"The copy of {info['vizier_id']} at {data_path} only contains "
                f"sources with {described}; fainter sources are missing.",
                CatalogStoreWarning,
                stacklevel=2,
            )
        return table

    def _info(self, catalog):
        """Return the loaded manifest for a catalog identifier or directory name."""
        catalog_id = self._ids_by_name.get(catalog, catalog)
        try:
            return self._catalogs[catalog_id]
        except KeyError:
            available = ", ".join(
                f"{catalog_id} ({name})"
                for name, catalog_id in sorted(self._ids_by_name.items())
            )
            raise CatalogNotFoundError(
                f"No catalog {catalog!r} in {self.root}; "
                f"it has: {available or 'no catalogs'}"
            ) from None


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _load_manifest(manifest):
    """
    Read and check one ``catalog.json``.

    Parameters
    ----------
    manifest : `pathlib.Path`
        Path to the manifest.

    Returns
    -------
    dict
        The manifest, plus three derived entries whose names start with an
        underscore: ``_path`` (the data directory), ``_zones`` (the zones the
        store holds) and ``_zones_recorded`` (whether the manifest listed
        them).

    Raises
    ------
    OSError, ValueError, KeyError, TypeError
        If the file cannot be read, is not JSON, is not a manifest of this
        format, is from a later format version or lacks a required key.
    """
    info = json.loads(manifest.read_text())
    if not isinstance(info, dict):
        raise TypeError(f"manifest is a {type(info).__name__}, not a dict")
    if info.get("format") != STORE_FORMAT:
        raise ValueError(f"format is {info.get('format')!r}, not {STORE_FORMAT!r}")
    # Manifests from before the version was recorded are version 1.
    version = info.get("format_version", 1)
    if version != STORE_FORMAT_VERSION:
        raise ValueError(
            f"format_version is {version!r}; this reader understands "
            f"{STORE_FORMAT_VERSION}"
        )
    for key in ("vizier_id", "ra_col", "dec_col"):
        if key not in info:
            raise KeyError(key)
    # The data lives in a sub-directory named by the manifest, so a rebuild
    # can publish a new one by rewriting only the manifest.
    info["_path"] = manifest.parent / info.get("data_dir", "data")
    info["_zones"], info["_zones_recorded"] = _store_zones(info)
    return info


def _store_zones(info):
    """
    Return the declination zones a store holds and whether its manifest says so.

    Parameters
    ----------
    info : dict
        The catalog's manifest, with ``_path`` set to its data directory.

    Returns
    -------
    zones : frozenset of int
        Zone indices, from the manifest's ``zones`` if present, otherwise from
        the ``zone=*`` directories on disk.
    recorded : bool
        ``True`` if ``zones`` came from the manifest.
    """
    if info.get("zones") is not None:
        return frozenset(int(z) for z in info["zones"]), True
    # Manifests written before zones were recorded: list the disk once.
    path = info["_path"]
    if not path.is_dir():
        return frozenset(), False
    zones = set()
    for zone_dir in path.glob("zone=*"):
        try:
            zones.add(int(zone_dir.name.partition("=")[2]))
        except ValueError:
            continue
    return frozenset(zones), False


def _missing_zones(info, zone_lo, zone_hi):
    """
    Return the zones in ``zone_lo..zone_hi`` the store should have but lacks.

    Parameters
    ----------
    info : dict
        The catalog's manifest, as returned by `_load_manifest`.
    zone_lo, zone_hi : int
        First and last zone a query needs.

    Returns
    -------
    list of int
        Missing zones, in order; empty if nothing is missing.

    Notes
    -----
    Zones outside the lowest..highest written zone are never missing: the
    store was simply built for less of the sky. When the manifest records
    its zones, a needed zone it lists must exist on disk, and a zone it does
    not list had no rows. When the zones were found on disk (older
    manifests), any gap inside the written range counts as missing: on a
    full-sky store every zone has stars, so a gap means lost data.
    """
    zones = info["_zones"]
    if not zones:
        return []
    needed = range(max(zone_lo, min(zones)), min(zone_hi, max(zones)) + 1)
    if info["_zones_recorded"]:
        path = info["_path"]
        return [z for z in needed if z in zones and not (path / f"zone={z}").is_dir()]
    return [z for z in needed if z not in zones]


def _covers(info, column_filters):
    """
    Return ``True`` if the store holds every source the filters could select.

    Parameters
    ----------
    info : dict
        The catalog's manifest.
    column_filters : dict
        The query's filters.

    Returns
    -------
    bool
        ``True`` if the manifest records no ``mag_limit``, or if one of the
        filters is an upper bound (``<`` or ``<=``) on a limited column at or
        brighter than the store's limit. See `CatalogStore.is_complete`.
    """
    limits = info.get("mag_limit") or {}
    if not limits:
        return True
    for col, spec in column_filters.items():
        if col in limits:
            op, a, _ = _parse_filter(spec)
            if op in ("<=", "<") and a <= limits[col]:
                return True
    return False


# --------------------------------------------------------------------------
# Column filters
# --------------------------------------------------------------------------

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_FILTER_RE = re.compile(
    rf"^\s*(<=|>=|!=|<|>|=)?\s*({_NUMBER})\s*(?:\.\.\s*({_NUMBER}))?\s*$"
)


def _parse_filter(spec):
    """
    Parse one column filter.

    Parameters
    ----------
    spec : str
        ``<=x``, ``<x``, ``>=x``, ``>x``, ``=x``, ``x``, ``!=x`` or ``a..b``.

    Returns
    -------
    op : str
        The comparison, with ``"="`` for a bare number and ``".."`` for a
        range.
    a : float
        The limit, or the lower end of a range.
    b : float or None
        The upper end of a range; `None` otherwise.

    Raises
    ------
    ValueError
        If ``spec`` is not one of the supported forms.
    """
    match = _FILTER_RE.match(str(spec))
    if not match:
        raise ValueError(f"Unsupported column filter {spec!r}")
    op, a, b = match.groups()
    if b is not None:
        return "..", float(a), float(b)
    return op or "=", float(a), None


def _filter_expression(field_expr, spec, arrow_type):
    """
    Translate one column filter into a pyarrow compute expression.

    Parameters
    ----------
    field_expr : `pyarrow.compute.Expression`
        The column to filter, e.g. ``pyarrow.compute.field("rmag")``.
    spec : str
        Filter in the syntax `_parse_filter` accepts, e.g. ``"<=15"`` or
        ``"10..12"``.
    arrow_type : `pyarrow.DataType`
        Type of the column in the dataset.

    Returns
    -------
    `pyarrow.compute.Expression` or None
        Expression that is true for rows passing the filter, or `None` if the
        filter cannot be expressed for a column of this type.

    Raises
    ------
    ValueError
        If ``spec`` is not a supported filter.

    Notes
    -----
    The expression selects the same rows as `_apply_column_filters`, so it can
    be pushed into a Parquet scan with that function still applied afterwards
    as the authority on semantics:

    * Floating-point limits are made scalars of the column's own type, so a
      float32 value sitting exactly at a limit is kept, as it is client-side.
    * Integer and boolean columns are compared as float64, matching the
      client-side conversion to float.
    * A comparison with a null is null, and ``filter`` drops null rows, so a
      null (masked) value never passes.
    * Other types (strings, for example) give `None`; the client-side filter
      then decides what to do with them.
    """
    op, a, b = _parse_filter(spec)
    if pa.types.is_floating(arrow_type):
        scalar_type = arrow_type
    elif pa.types.is_integer(arrow_type) or pa.types.is_boolean(arrow_type):
        field_expr = field_expr.cast(pa.float64())
        scalar_type = pa.float64()
    else:
        return None
    a = pa.scalar(a, type=scalar_type)
    if op == "..":
        return (field_expr >= a) & (field_expr <= pa.scalar(b, type=scalar_type))
    return {
        "<=": field_expr <= a,
        "<": field_expr < a,
        ">=": field_expr >= a,
        ">": field_expr > a,
        "=": field_expr == a,
        "!=": field_expr != a,
    }[op]


def _apply_column_filters(table, column_filters):
    """
    Apply column filters to an astropy table.

    Parameters
    ----------
    table : `astropy.table.Table`
        Rows to filter.
    column_filters : dict
        Filters keyed by column name, in the syntax `_parse_filter` accepts.

    Returns
    -------
    `astropy.table.Table`
        The rows that pass every filter. A masked value never passes.

    Raises
    ------
    ValueError
        If a filter is malformed or names a column the table lacks.
    """
    if not column_filters:
        return table
    keep = np.ones(len(table), dtype=bool)
    for col, spec in column_filters.items():
        if col not in table.colnames:
            raise ValueError(f"Cannot filter on {col!r}: column not in catalog")
        op, a, b = _parse_filter(spec)
        values = np.ma.asarray(table[col])
        if values.dtype.kind == "f":
            # Compare in the column's own precision so a star sitting exactly
            # at a limit is kept (a float32 15.123 is below a float64 15.123).
            a = values.dtype.type(a)
            b = values.dtype.type(b) if b is not None else None
        else:
            values = np.ma.asarray(values, dtype=float)
        if op == "..":
            ok = (values >= a) & (values <= b)
        else:
            ok = {
                "<=": values <= a,
                "<": values < a,
                ">=": values >= a,
                ">": values > a,
                "=": values == a,
                "!=": values != a,
            }[op]
        keep &= np.ma.filled(ok, False)
    return table[keep]


# --------------------------------------------------------------------------
# Geometry and conversion
# --------------------------------------------------------------------------


def _angular_separation_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees, inputs in degrees."""
    ra1, dec1, ra2, dec2 = (
        np.radians(np.asarray(x, dtype=float)) for x in (ra1, dec1, ra2, dec2)
    )
    return np.degrees(angular_separation(ra1, dec1, ra2, dec2))


def _arrow_to_masked_table(arrow_table, units):
    """
    Convert a pyarrow table to a masked astropy table.

    Parameters
    ----------
    arrow_table : `pyarrow.Table`
        Rows read from a store.
    units : dict
        Unit string for each column that has one, keyed by column name.

    Returns
    -------
    `astropy.table.Table`
        Masked table with the same columns, in the same order.

    Notes
    -----
    Every column becomes a `~astropy.table.MaskedColumn` whose mask is the
    Arrow null mask. Values under the mask are filled before conversion, with
    ``0`` (``False`` for booleans), because pyarrow would otherwise turn an
    integer or boolean column with nulls into float64, changing the dtype and
    rounding large integers.

    String columns become fixed-width numpy unicode columns, with nulls filled
    with the empty string under the mask.
    """
    table = Table(masked=True)
    for name in arrow_table.column_names:
        column = arrow_table[name]
        mask = column.is_null().to_numpy(zero_copy_only=False)
        if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
            values = np.array(column.fill_null("").to_pylist(), dtype=str)
        else:
            fill = False if pa.types.is_boolean(column.type) else 0
            values = column.fill_null(fill).to_numpy(zero_copy_only=False)
        table[name] = MaskedColumn(values, mask=mask, unit=units.get(name))
    return table
