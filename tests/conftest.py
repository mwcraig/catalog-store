"""
Shared test helpers.

``write_store`` is a deliberately small writer used to build stores with
particular shapes (three zones, stars at the poles, a shallow store...). It
follows ``docs/store-format.md`` and is NOT the production writer: real stores
are written by stellarphot's ``write_local_store``. The golden store in
``tests/data/store`` was written by that function, so the reader is also
tested against the real thing.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

DATA = Path(__file__).parent / "data"
REFCAT2_ID = "J/ApJ/867/105/refcat2"
STARS_ID = "test/stars"


def write_store(
    table,
    root,
    name,
    catalog_id,
    ra_col,
    dec_col,
    mag_limit=None,
    zone_deg=1.0,
    record_zones=True,
):
    """Write ``table`` to ``root / name`` as a store and return that path."""
    cat_dir = Path(root) / name
    data_dir = cat_dir / "data-test"
    shutil.rmtree(cat_dir, ignore_errors=True)
    data_dir.mkdir(parents=True)

    columns = {}
    units = {}
    for colname in table.colnames:
        column = table[colname]
        if column.unit is not None:
            units[colname] = column.unit.to_string()
        values = np.ma.asarray(column)
        mask = np.ma.getmaskarray(values)
        data = np.ma.getdata(values)
        if data.dtype.kind == "U":
            columns[colname] = pa.array(
                [None if m else str(v) for v, m in zip(data, mask, strict=True)],
                type=pa.string(),
            )
        else:
            columns[colname] = pa.array(data, mask=mask)
    dec = np.asarray(table[dec_col], dtype=float)
    zone = np.floor((dec + 90) / zone_deg).astype(np.int16)
    columns["zone"] = pa.array(zone, type=pa.int16())
    arrow_table = pa.table(columns).sort_by(
        [("zone", "ascending"), (ra_col, "ascending")]
    )
    ds.write_dataset(
        arrow_table,
        data_dir,
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("zone", pa.int16())]), flavor="hive"),
    )
    manifest = dict(
        format="stellarphot-catalog-store",
        format_version=1,
        vizier_id=catalog_id,
        name=name,
        layout="dec_zone_parquet",
        zone_deg=zone_deg,
        ra_col=ra_col,
        dec_col=dec_col,
        data_dir=data_dir.name,
        rows=len(table),
        version="written by the tests",
        mag_limit=mag_limit,
        units=units,
    )
    if record_zones:
        manifest["zones"] = sorted(int(z) for z in np.unique(zone))
    (cat_dir / "catalog.json").write_text(json.dumps(manifest, indent=1))
    return cat_dir


def read_manifest(root, name):
    return json.loads((Path(root) / name / "catalog.json").read_text())


def data_dir(root, name):
    return Path(root) / name / read_manifest(root, name)["data_dir"]


def stars(names, ras, decs):
    return Table({"Name": names, "RAJ2000": ras, "DEJ2000": decs})


def write_stars(table, root, **kwargs):
    return write_store(table, root, "stars", STARS_ID, "RAJ2000", "DEJ2000", **kwargs)


def star_names(store, ra, dec, radius):
    table = store.cone_search(STARS_ID, SkyCoord(ra, dec, unit="deg"), radius)
    return set(table["Name"])


def refcat2_table():
    """The rows in the golden store, as written (a few values are masked)."""
    return Table.read(DATA / "refcat2_ey_uma.ecsv")


def assert_tables_equal(expected_table, table):
    """Assert same columns, rows in the same order, masks, values and units."""
    assert table.colnames == expected_table.colnames
    assert len(table) == len(expected_table)
    for colname in expected_table.colnames:
        expected = np.ma.asarray(expected_table[colname])
        got = np.ma.asarray(table[colname])
        np.testing.assert_array_equal(
            np.ma.getmaskarray(expected), np.ma.getmaskarray(got), err_msg=colname
        )
        if expected.dtype.kind in "fiu":
            np.testing.assert_array_equal(
                expected.filled(0), got.filled(0), err_msg=colname
            )
        else:
            assert list(expected.filled("")) == list(got.filled("")), colname
        assert table[colname].unit == expected_table[colname].unit, colname


@pytest.fixture
def ey_uma():
    """Center of the field the golden store covers."""
    return SkyCoord(135.55089614, 49.84289679, unit="deg")


@pytest.fixture
def golden_root(tmp_path):
    """A scratch copy of the golden store, safe for a test to damage."""
    root = tmp_path / "store"
    shutil.copytree(DATA / "store", root)
    return root


@pytest.fixture
def shallow_root(tmp_path):
    """The golden rows in a store that records it only reaches rmag 16."""
    root = tmp_path / "shallow"
    write_store(
        refcat2_table(),
        root,
        "refcat2",
        REFCAT2_ID,
        "RA_ICRS",
        "DE_ICRS",
        mag_limit={"rmag": 16.0},
    )
    return root
