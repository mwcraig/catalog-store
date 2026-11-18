"""
Building a refcat2 store from MAST tiles.

``tests/data/tiles`` holds excerpts of eight real tiles (two tiles from each
of the four magnitude bins to m < 19), cut down to the field of EY UMa. The
golden table ``refcat2_ey_uma.ecsv`` is what VizieR returned for the same
field, so building from the excerpts must reproduce VizieR's values.
"""

import shutil

import numpy as np
import pytest
from astropy import units as u
from conftest import DATA, REFCAT2_ID, data_dir, read_manifest, refcat2_table

from catalog_store_reader import CatalogStore, CatalogStoreWarning
from catalog_store_reader.refcat2 import build_refcat2_store, main

TILES = DATA / "tiles"
BINS = ["00_m_16", "16_m_17", "17_m_18", "18_m_19"]
N_TILE_ROWS = 252  # rows in the eight tile excerpts

# Columns the golden table has masked by hand (see make_golden_store.py), so
# their masks say nothing about what VizieR blanks.
_HAND_MASKED = {"e_pmRA", "e_pmDE", "e_Gmag", "pmRA"}


def _build(root, bins=BINS, tiles=TILES, **kwargs):
    kwargs.setdefault("dec_min", 49)
    kwargs.setdefault("dec_max", 50)
    kwargs.setdefault("require_all_tiles", False)
    kwargs.setdefault("progress", None)
    return build_refcat2_store([tiles / b for b in bins], root, **kwargs)


def _whole_field(root, center):
    """Every row of the built store; with no filter it warns about its depth."""
    with pytest.warns(CatalogStoreWarning, match="fainter sources are missing"):
        return CatalogStore(root).cone_search(REFCAT2_ID, center, 1.0 * u.deg)


def _match_by_position(expected, table):
    """Row of ``table`` at exactly the position of each row of ``expected``."""
    by_position = {
        (ra, dec): i
        for i, (ra, dec) in enumerate(
            zip(table["RA_ICRS"], table["DE_ICRS"], strict=True)
        )
    }
    return [
        by_position.get((ra, dec))
        for ra, dec in zip(expected["RA_ICRS"], expected["DE_ICRS"], strict=True)
    ]


def test_build_reproduces_vizier_values(tmp_path, ey_uma):
    cat_dir = _build(tmp_path)
    assert cat_dir == tmp_path / "refcat2"
    built = _whole_field(tmp_path, ey_uma)
    assert len(built) == N_TILE_ROWS

    vizier = refcat2_table()
    brightest = np.minimum.reduce(
        [np.ma.filled(vizier[c], 99) for c in ("gmag", "rmag", "imag")]
    )
    vizier = vizier[brightest < 19]
    assert len(vizier) > 100

    rows = _match_by_position(vizier, built)
    assert None not in rows
    built = built[rows]
    # VizieR adds a constant "GaiaDR2" link label that is not in the tiles.
    assert set(vizier["GaiaDR2"]) == {"GaiaDR2"}
    vizier.remove_column("GaiaDR2")
    assert built.colnames == vizier.colnames
    for colname in vizier.colnames:
        assert built[colname].dtype == vizier[colname].dtype, colname
        assert built[colname].unit == vizier[colname].unit, colname
        expected = np.ma.asarray(vizier[colname])
        got = np.ma.asarray(built[colname])
        if colname not in _HAND_MASKED:
            np.testing.assert_array_equal(
                np.ma.getmaskarray(got), np.ma.getmaskarray(expected), err_msg=colname
            )
        compare = ~np.ma.getmaskarray(expected)
        assert not np.ma.getmaskarray(got)[compare].any(), colname
        np.testing.assert_array_equal(
            np.ma.getdata(got)[compare], np.ma.getdata(expected)[compare], colname
        )


def test_build_manifest(tmp_path):
    _build(tmp_path)
    manifest = read_manifest(tmp_path, "refcat2")
    assert manifest["format"] == "stellarphot-catalog-store"
    assert manifest["format_version"] == 1
    assert manifest["vizier_id"] == REFCAT2_ID
    assert manifest["ra_col"] == "RA_ICRS"
    assert manifest["dec_col"] == "DE_ICRS"
    assert manifest["rows"] == N_TILE_ROWS
    assert manifest["zones"] == [139, 140]
    assert manifest["mag_limit"] == {"gmag": 19.0, "rmag": 19.0, "imag": 19.0}
    assert manifest["units"]["rmag"] == "mag"
    assert manifest["units"]["pmRA"] == "mas / yr"
    for bin_name in BINS:
        assert bin_name in manifest["version"]
    # One part per (zone, bin).
    parts = sorted(
        p.name for p in data_dir(tmp_path, "refcat2").glob("zone=139/*.parquet")
    )
    assert len(parts) == 4
    assert all(b in name for b, name in zip(BINS, parts, strict=True))


def test_mag_limit_defaults_to_the_faintest_bin(tmp_path, ey_uma):
    _build(tmp_path, bins=BINS[:1])
    assert read_manifest(tmp_path, "refcat2")["mag_limit"] == {
        "gmag": 16.0,
        "rmag": 16.0,
        "imag": 16.0,
    }
    store = CatalogStore(tmp_path)
    assert store.is_complete(REFCAT2_ID, {"rmag": "<=15"})
    assert not store.is_complete(REFCAT2_ID, {"rmag": "<=17"})
    bright = store.cone_search(REFCAT2_ID, ey_uma, 1.0, {"rmag": "<=15"})
    assert 0 < len(bright) < 73


def test_mag_limit_can_be_brighter_than_the_faintest_bin(tmp_path):
    _build(tmp_path, bins=BINS[:2], mag_limit=16.5)
    assert read_manifest(tmp_path, "refcat2")["mag_limit"]["rmag"] == 16.5


@pytest.mark.parametrize(
    "bins, kwargs, message",
    [
        (["00_m_16", "17_m_18"], {}, "not contiguous"),
        (["16_m_17", "17_m_18"], {}, "start at 00_m_16"),
        (["00_m_16"], {"mag_limit": 17.0}, "exceeds the faintest bin"),
    ],
)
def test_bins_must_cover_the_store_depth(tmp_path, bins, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, bins=bins, **kwargs)
    assert not (tmp_path / "refcat2" / "catalog.json").exists()


def test_tile_directory_must_be_named_like_a_bin(tmp_path):
    bad = tmp_path / "my-tiles"
    shutil.copytree(TILES / "00_m_16", bad)
    with pytest.raises(ValueError, match="not a MAST refcat2 bin"):
        build_refcat2_store([bad], tmp_path / "out", progress=None)


def test_incomplete_bin_is_refused_by_default(tmp_path):
    with pytest.raises(ValueError, match="expected 64800 tiles, found 2"):
        _build(tmp_path, bins=BINS[:1], require_all_tiles=True)
    assert not (tmp_path / "refcat2" / "catalog.json").exists()


def test_empty_tile_is_skipped(tmp_path):
    # MAST has tiles with no stars in a bin: zero-length files.
    tiles = tmp_path / "tiles"
    shutil.copytree(TILES, tiles)
    (tiles / "00_m_16" / "136+49.rc2").write_text("")
    _build(tmp_path / "out", tiles=tiles)
    assert read_manifest(tmp_path / "out", "refcat2")["rows"] == N_TILE_ROWS


def test_resume_rebuilds_only_unfinished_zones(tmp_path, ey_uma):
    _build(tmp_path)
    data = data_dir(tmp_path, "refcat2")
    finished = sorted(data.glob("zone=140/*.parquet"))
    stamps = [p.stat().st_mtime_ns for p in finished]
    # Zone 139 lost a part, as if the build died while writing it; the
    # manifest is written last, so it would not exist either.
    sorted(data.glob("zone=139/*.parquet"))[-1].unlink()
    (tmp_path / "refcat2" / "catalog.json").unlink()

    _build(tmp_path, resume=True)
    assert len(list(data.glob("zone=139/*.parquet"))) == 4
    assert [p.stat().st_mtime_ns for p in finished] == stamps
    assert read_manifest(tmp_path, "refcat2")["rows"] == N_TILE_ROWS
    assert read_manifest(tmp_path, "refcat2")["zones"] == [139, 140]
    assert len(_whole_field(tmp_path, ey_uma)) == N_TILE_ROWS


def test_fresh_build_unpublishes_the_old_store_first(tmp_path):
    _build(tmp_path)
    # A build that fails part-way must not leave the old manifest pointing at
    # half-replaced data.
    with pytest.raises(ValueError, match="expected 64800"):
        _build(tmp_path, require_all_tiles=True)
    # Refused before touching anything: the old store is intact.
    assert read_manifest(tmp_path, "refcat2")["rows"] == N_TILE_ROWS

    tiles = tmp_path / "tiles"
    shutil.copytree(TILES, tiles)
    (tiles / "17_m_18" / "135+49.rc2").write_text("not,a,tile\n")
    with pytest.raises(Exception):  # noqa: B017 - whatever the CSV reader raises
        _build(tmp_path, tiles=tiles)
    assert not (tmp_path / "refcat2" / "catalog.json").exists()


def test_command_line(tmp_path, capsys):
    argv = ["--out", str(tmp_path), "--dec-min", "49", "--dec-max", "50"]
    argv += ["--allow-missing-tiles", "--mag-limit", "18.5", "--name", "rc2"]
    for bin_name in BINS:
        argv += ["--tiles", str(TILES / bin_name)]
    assert main(argv) is None
    manifest = read_manifest(tmp_path, "rc2")
    assert manifest["rows"] == N_TILE_ROWS
    assert manifest["mag_limit"]["gmag"] == 18.5
    assert manifest["name"] == "rc2"
    assert f"{N_TILE_ROWS} rows" in capsys.readouterr().out
