"""
Write a catalog store that `~catalog_store_reader.CatalogStore` can read.

`write_store` takes a catalog as an astropy table, or as an iterable of
tables for a catalog too large to hold in memory. ``docs/store-format.md``
in the repository describes what it writes.

The code was extracted from ``stellarphot.catalog_backends.write_local_store``
(https://github.com/feder-observatory/stellarphot).
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from astropy.table import Table

from .reader import STORE_FORMAT, STORE_FORMAT_VERSION

__all__ = ["write_store"]

# Small row groups, sorted by right ascension, are what let a cone search
# skip most of a zone using the row-group statistics.
_ROWS_PER_GROUP = 20_000


def write_store(
    table,
    root,
    name,
    catalog_id,
    ra_col,
    dec_col,
    version=None,
    mag_limit=None,
    zone_deg=1.0,
):
    """
    Write a catalog as a store that `~catalog_store_reader.CatalogStore` reads.

    Parameters
    ----------
    table : `astropy.table.Table` or iterable of tables
        Catalog rows, with masked values where the catalog has blanks. An
        iterable of tables with the same columns is written chunk by chunk.
    root : str or `pathlib.Path`
        Store directory; the catalog goes in ``root / name``.
    name : str
        Short directory name, e.g. ``"refcat2"``.
    catalog_id : str
        Identifier the catalog is asked for by, e.g. its VizieR identifier
        ``"J/ApJ/867/105/refcat2"``. Stored in the manifest as ``vizier_id``.
    ra_col, dec_col : str
        Names of the coordinate columns, which must be in ICRS degrees.
    version : str, optional
        Free-form provenance string recorded in the manifest.
    mag_limit : dict, optional
        ``{column: limit}`` describing how deep the store is complete, e.g.
        ``{"rmag": 16.0}``: every source at or brighter than the limit in at
        least one of the columns is present. Leave out for a whole catalog.
    zone_deg : float, optional
        Height of the declination zones the data is partitioned into.

    Returns
    -------
    `pathlib.Path`
        Path to the catalog directory.

    Notes
    -----
    A rebuild writes into a new ``data-<token>`` directory and switches the
    manifest's ``data_dir`` to it only once every chunk is written, so an
    interrupted rebuild leaves the published store intact. Older data
    directories are removed after the switch.

    Each chunk becomes its own Parquet file in every zone it has rows in, so
    many small chunks give many small files. Chunks of a few million rows
    work well.
    """
    cat_dir = Path(root) / name
    cat_dir.mkdir(parents=True, exist_ok=True)
    data_name = f"data-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    data_dir = cat_dir / data_name
    data_dir.mkdir()

    chunks = [table] if isinstance(table, Table) else table
    units = {}
    zones = set()
    rows = 0
    try:
        for chunk_number, chunk in enumerate(chunks):
            arrow_table = _table_to_arrow(chunk, ra_col, dec_col, zone_deg, units)
            zones.update(arrow_table["zone"].unique().to_pylist())
            _write_parts(arrow_table, data_dir, f"part-{chunk_number}")
            rows += len(chunk)
    except BaseException:
        # Leave the published store as it was.
        shutil.rmtree(data_dir, ignore_errors=True)
        raise
    _write_manifest(
        cat_dir,
        catalog_id=catalog_id,
        name=name,
        zone_deg=zone_deg,
        ra_col=ra_col,
        dec_col=dec_col,
        data_dir=data_name,
        rows=rows,
        version=version,
        mag_limit=mag_limit,
        units=units,
        zones=zones,
    )
    for old in [cat_dir / "data", *cat_dir.glob("data-*")]:
        if old != data_dir and old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
    return cat_dir


def _table_to_arrow(table, ra_col, dec_col, zone_deg, units):
    """
    Convert catalog rows to a pyarrow table ready to be written.

    Parameters
    ----------
    table : `astropy.table.Table`
        Catalog rows.
    ra_col, dec_col : str
        Names of the coordinate columns, in degrees.
    zone_deg : float
        Height of the declination zones.
    units : dict
        Updated in place with the unit string of each column that has a unit.

    Returns
    -------
    `pyarrow.Table`
        The rows plus an int16 ``zone`` column, sorted by zone and then by
        right ascension. Masked values are nulls.
    """
    columns = {}
    for colname in table.colnames:
        column = table[colname]
        # Read the unit once: see the test that guards this.
        unit = column.unit
        if unit is not None:
            units[colname] = unit.to_string()
        values = np.ma.asarray(column)
        mask = np.ma.getmaskarray(values)
        data = np.ma.getdata(values)
        if data.dtype.kind in ("U", "S", "O"):
            # Byte strings (FITS text columns) are decoded so they are stored
            # as text, not as their repr, "b'...'".
            texts = [v.decode() if isinstance(v, bytes) else str(v) for v in data]
            columns[colname] = pa.array(
                [None if m else t for t, m in zip(texts, mask, strict=True)],
                type=pa.string(),
            )
        else:
            # pyarrow only takes native byte order; FITS is big-endian.
            if not data.dtype.isnative:
                data = data.astype(data.dtype.newbyteorder("="), copy=False)
            columns[colname] = pa.array(data, mask=mask)
    # The zone column only drives the partitioning; the reader drops it.
    dec = np.asarray(table[dec_col], dtype=float)
    zone = np.floor((dec + 90) / zone_deg).astype(np.int16)
    columns["zone"] = pa.array(zone, type=pa.int16())
    return pa.table(columns).sort_by([("zone", "ascending"), (ra_col, "ascending")])


def _write_parts(arrow_table, data_dir, basename):
    """
    Write rows into ``data_dir/zone=N/<basename>-<i>.parquet``.

    Parameters
    ----------
    arrow_table : `pyarrow.Table`
        Rows with a ``zone`` column, as returned by `_table_to_arrow`.
    data_dir : `pathlib.Path`
        The catalog's data directory.
    basename : str
        Start of the file name; it must differ between calls that write to
        the same zone, or the later call replaces the earlier file.
    """
    ds.write_dataset(
        arrow_table,
        data_dir,
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("zone", pa.int16())]), flavor="hive"),
        basename_template=basename + "-{i}.parquet",
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        max_rows_per_group=_ROWS_PER_GROUP,
        existing_data_behavior="overwrite_or_ignore",
    )


def _write_manifest(
    cat_dir,
    *,
    catalog_id,
    name,
    zone_deg,
    ra_col,
    dec_col,
    data_dir,
    rows,
    version,
    mag_limit,
    units,
    zones,
):
    """
    Publish a catalog by writing its ``catalog.json`` in one step.

    Parameters
    ----------
    cat_dir : `pathlib.Path`
        The catalog directory.
    catalog_id, name, zone_deg, ra_col, dec_col, data_dir, rows, version, \
mag_limit, units, zones
        The manifest's contents; see ``docs/store-format.md``.
    """
    manifest = dict(
        format=STORE_FORMAT,
        format_version=STORE_FORMAT_VERSION,
        vizier_id=catalog_id,
        name=name,
        layout="dec_zone_parquet",
        zone_deg=zone_deg,
        ra_col=ra_col,
        dec_col=dec_col,
        data_dir=data_dir,
        rows=rows,
        version=version,
        mag_limit=mag_limit,
        units=units,
        zones=sorted(int(z) for z in zones),
    )
    # Write beside the manifest and rename, so a reader never sees half a file.
    manifest_tmp = cat_dir / "catalog.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(manifest_tmp, cat_dir / "catalog.json")
