import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, MaskedColumn, Table
from conftest import (
    DATA,
    REFCAT2_ID,
    STARS_ID,
    assert_tables_equal,
    data_dir,
    read_manifest,
    refcat2_table,
    star_names,
    stars,
    write_stars,
)

from catalog_store import CatalogStore, write_store


def _write_refcat2(root, **kwargs):
    return write_store(
        refcat2_table(), root, "refcat2", REFCAT2_ID, "RA_ICRS", "DE_ICRS", **kwargs
    )


def test_layout_and_manifest(tmp_path):
    cat_dir = _write_refcat2(tmp_path, version="a test", mag_limit={"rmag": 16.0})
    assert cat_dir == tmp_path / "refcat2"
    manifest = read_manifest(tmp_path, "refcat2")
    assert manifest["format"] == "stellarphot-catalog-store"
    assert manifest["format_version"] == 1
    assert manifest["vizier_id"] == REFCAT2_ID
    assert manifest["name"] == "refcat2"
    assert manifest["ra_col"] == "RA_ICRS"
    assert manifest["dec_col"] == "DE_ICRS"
    assert manifest["zone_deg"] == 1.0
    assert manifest["rows"] == len(refcat2_table())
    assert manifest["version"] == "a test"
    assert manifest["mag_limit"] == {"rmag": 16.0}
    assert manifest["units"]["pmRA"] == "mas / yr"
    assert manifest["units"]["RA_ICRS"] == "deg"
    # The field straddles Dec = +50, so the rows land in two zones.
    assert manifest["zones"] == [139, 140]
    parts = sorted(
        p.parent.name for p in data_dir(tmp_path, "refcat2").glob("zone=*/*.parquet")
    )
    assert parts == ["zone=139", "zone=140"]


def test_written_store_reads_back_as_written(tmp_path, ey_uma):
    _write_refcat2(tmp_path)
    expected = refcat2_table()
    table = CatalogStore(tmp_path).cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg)
    table.sort("RA_ICRS")
    expected.sort("RA_ICRS")
    assert_tables_equal(expected, table)


def test_written_store_matches_the_one_stellarphot_wrote(tmp_path, ey_uma):
    _write_refcat2(tmp_path)
    ours = CatalogStore(tmp_path).cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg)
    theirs = CatalogStore(DATA / "store").cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg)
    ours.sort("RA_ICRS")
    theirs.sort("RA_ICRS")
    assert_tables_equal(theirs, ours)
    for colname in theirs.colnames:
        assert ours[colname].dtype == theirs[colname].dtype, colname
    ours_manifest = read_manifest(tmp_path, "refcat2")
    theirs_manifest = read_manifest(DATA / "store", "refcat2")
    for key in theirs_manifest:
        if key not in ("data_dir", "version"):
            assert ours_manifest[key] == theirs_manifest[key], key


def test_rows_are_sorted_by_ra_within_each_file(tmp_path):
    import pyarrow.parquet as pq

    _write_refcat2(tmp_path)
    files = sorted(data_dir(tmp_path, "refcat2").glob("zone=*/*.parquet"))
    assert files
    for path in files:
        ra = pq.read_table(path, columns=["RA_ICRS"])["RA_ICRS"].to_numpy()
        assert (np.diff(ra) >= 0).all()
        # The zone is in the directory name only.
        assert "zone" not in pq.read_schema(path).names


def test_chunks_are_written_one_at_a_time(tmp_path):
    def chunks():
        yield stars(["a", "b"], [30.0, 31.0], [10.5, 11.5])
        yield stars(["c"], [30.5], [10.6])

    write_stars(chunks(), tmp_path)
    manifest = read_manifest(tmp_path, "stars")
    assert manifest["rows"] == 3
    assert manifest["zones"] == [100, 101]
    assert star_names(CatalogStore(tmp_path), 30.5, 11.0, 2.0) == {"a", "b", "c"}


def test_rebuild_replaces_data(tmp_path):
    write_stars(stars(["old"], [30.0], [10.5]), tmp_path)
    old_path = data_dir(tmp_path, "stars")
    assert old_path.is_dir()

    write_stars(stars(["new"], [30.0], [10.5]), tmp_path)
    new_path = data_dir(tmp_path, "stars")
    assert new_path != old_path
    assert not old_path.exists()
    assert [p.name for p in (tmp_path / "stars").glob("data*")] == [new_path.name]
    assert star_names(CatalogStore(tmp_path), 30.0, 10.5, 0.2) == {"new"}


def test_failed_rebuild_keeps_old_store(tmp_path):
    write_stars(stars(["old"], [30.0], [10.5]), tmp_path)
    manifest_before = (tmp_path / "stars" / "catalog.json").read_text()
    old_path = data_dir(tmp_path, "stars")
    store = CatalogStore(tmp_path)

    def chunks():
        yield stars(["new"], [30.0], [10.5])
        raise RuntimeError("source catalog went away")

    with pytest.raises(RuntimeError, match="went away"):
        write_stars(chunks(), tmp_path)
    assert (tmp_path / "stars" / "catalog.json").read_text() == manifest_before
    # The partial new data directory is cleaned up.
    assert [p.name for p in (tmp_path / "stars").glob("data*")] == [old_path.name]
    assert star_names(store, 30.0, 10.5, 0.2) == {"old"}


def test_big_endian_and_bytes_columns(tmp_path):
    # Tables read from FITS have big-endian numbers and byte-string text.
    table = Table(
        [
            Column(np.array([b"EY UMa", b"star 2"], dtype="S"), name="Name"),
            Column(np.array([30.0, 30.1], dtype=">f8"), name="RAJ2000"),
            Column(np.array([10.5, 10.6], dtype=">f8"), name="DEJ2000"),
            MaskedColumn(
                np.array([12.5, 13.5], dtype=">f4"), mask=[False, True], name="Vmag"
            ),
            Column(np.array([7, 2**40], dtype=">i8"), name="id"),
        ]
    )
    write_stars(table, tmp_path)
    got = CatalogStore(tmp_path).cone_search(
        STARS_ID, SkyCoord(30.0, 10.5, unit="deg"), 1.0
    )
    got.sort("RAJ2000")
    assert list(got["Name"]) == ["EY UMa", "star 2"]
    for colname in ["RAJ2000", "DEJ2000", "Vmag", "id"]:
        np.testing.assert_array_equal(
            np.ma.getmaskarray(got[colname]), np.ma.getmaskarray(table[colname])
        )
        np.testing.assert_array_equal(
            np.ma.filled(got[colname], 0), np.ma.filled(table[colname], 0)
        )


class _UnitReadOnceColumn(Column):
    """Column whose unit reads as `None` after the first read once armed."""

    real_unit_reads = None  # None means not armed; the unit always reads.

    @property
    def unit(self):
        if self.real_unit_reads is None:
            return Column.unit.fget(self)
        self.real_unit_reads -= 1
        return Column.unit.fget(self) if self.real_unit_reads >= 0 else None

    @unit.setter
    def unit(self, unit):
        Column.unit.fset(self, unit)


def test_each_unit_is_read_once(tmp_path):
    """
    Guard against a unit seen as set and then as `None` within one chunk.

    One chunk in 720 of a 785-million-row build raised ``AttributeError``
    because ``column.unit`` was not `None` when checked and `None` when used.
    """
    table = Table(
        [
            _UnitReadOnceColumn([10.0, 20.0], name="RAJ2000", unit="deg"),
            Column([1.0, 2.0], name="DEJ2000", unit="deg"),
        ]
    )
    table["RAJ2000"].real_unit_reads = 1
    write_stars(table, tmp_path)
    assert read_manifest(tmp_path, "stars")["units"] == {
        "RAJ2000": "deg",
        "DEJ2000": "deg",
    }
