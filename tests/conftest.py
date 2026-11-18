"""
Shared test helpers.

Most test stores are written on the fly with this package's own
``write_store``. The golden store in ``tests/data/store`` was written by
stellarphot's ``write_local_store`` instead, so the reader is also tested
against a store this package did not write.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from catalog_store_reader import write_store

DATA = Path(__file__).parent / "data"
REFCAT2_ID = "J/ApJ/867/105/refcat2"
STARS_ID = "test/stars"


def read_manifest(root, name):
    return json.loads((Path(root) / name / "catalog.json").read_text())


def data_dir(root, name):
    return Path(root) / name / read_manifest(root, name)["data_dir"]


def stars(names, ras, decs):
    return Table({"Name": names, "RAJ2000": ras, "DEJ2000": decs})


def write_stars(table, root, record_zones=True, **kwargs):
    """
    Write ``table`` as the ``stars`` catalog in ``root``.

    With ``record_zones=False`` the zone list is removed from the manifest,
    as in stores written before zones were recorded.
    """
    cat_dir = write_store(
        table, root, "stars", STARS_ID, "RAJ2000", "DEJ2000", **kwargs
    )
    if not record_zones:
        manifest = read_manifest(root, "stars")
        del manifest["zones"]
        (cat_dir / "catalog.json").write_text(json.dumps(manifest))
    return cat_dir


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
