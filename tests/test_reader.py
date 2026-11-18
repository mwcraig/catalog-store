import json
import shutil
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
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

from catalog_store import (
    CatalogNotFoundError,
    CatalogStore,
    CatalogStoreWarning,
    CatalogUnavailableError,
)
from catalog_store.reader import _apply_column_filters

# ---------------------------------------------------------------------------
# Opening a store and finding catalogs in it
# ---------------------------------------------------------------------------


def test_store_lists_its_catalogs():
    store = CatalogStore(DATA / "store")
    assert store.catalog_ids == (REFCAT2_ID,)
    assert REFCAT2_ID in store
    assert "B/vsx/vsx" not in store
    manifest = store.manifest(REFCAT2_ID)
    assert manifest["rows"] == 187
    assert manifest["ra_col"] == "RA_ICRS"
    # A copy: changing it must not change the store.
    manifest["ra_col"] = "nope"
    assert store.manifest(REFCAT2_ID)["ra_col"] == "RA_ICRS"


def test_catalog_can_be_named_by_its_directory(ey_uma):
    store = CatalogStore(DATA / "store")
    assert "refcat2" in store
    by_id = store.cone_search(REFCAT2_ID, ey_uma, 0.2 * u.deg)
    by_name = store.cone_search("refcat2", ey_uma, 0.2 * u.deg)
    assert len(by_id) > 0
    assert_tables_equal(by_id, by_name)
    assert store.manifest("refcat2") == store.manifest(REFCAT2_ID)


def test_unknown_catalog_is_not_found(ey_uma):
    store = CatalogStore(DATA / "store")
    with pytest.raises(CatalogNotFoundError, match="II/336/apass9.*refcat2"):
        store.cone_search("II/336/apass9", ey_uma, 0.2 * u.deg)
    with pytest.raises(CatalogNotFoundError):
        store.manifest("apass")
    with pytest.raises(CatalogNotFoundError):
        store.is_complete("apass")


def test_store_warns_about_missing_or_empty_root(tmp_path):
    with pytest.warns(CatalogStoreWarning, match="does not exist"):
        store = CatalogStore(tmp_path / "typo")
    assert store.catalog_ids == ()
    with pytest.warns(CatalogStoreWarning, match="no catalogs"):
        CatalogStore(tmp_path)


@pytest.mark.parametrize(
    "manifest_text",
    [
        '{"format": "stellarphot-catalog-store", "vizier_id": "te',  # truncated
        json.dumps({"format": "something-else", "vizier_id": "test/other"}),
        json.dumps({"vizier_id": "test/other"}),  # no format
        json.dumps({"format": "stellarphot-catalog-store"}),  # no vizier_id
        json.dumps(["not", "a", "dict"]),
        json.dumps(  # written by a later, incompatible writer
            {
                "format": "stellarphot-catalog-store",
                "format_version": 2,
                "vizier_id": "test/other",
            }
        ),
    ],
)
def test_store_skips_bad_manifest(golden_root, ey_uma, manifest_text):
    bad = golden_root / "other" / "catalog.json"
    bad.parent.mkdir()
    bad.write_text(manifest_text)
    with pytest.warns(CatalogStoreWarning, match="other.catalog.json") as record:
        store = CatalogStore(golden_root)
    assert len(record) == 1
    assert store.catalog_ids == (REFCAT2_ID,)
    assert len(store.cone_search(REFCAT2_ID, ey_uma, 0.5 * u.deg)) > 0


def test_manifest_without_format_version_is_version_1(golden_root, ey_uma):
    manifest = read_manifest(golden_root, "refcat2")
    del manifest["format_version"]
    (golden_root / "refcat2" / "catalog.json").write_text(json.dumps(manifest))
    store = CatalogStore(golden_root)
    assert len(store.cone_search(REFCAT2_ID, ey_uma, 0.5 * u.deg)) > 0


# ---------------------------------------------------------------------------
# Cone searches on the golden store (written by stellarphot's real writer)
# ---------------------------------------------------------------------------


def test_cone_over_whole_field_returns_every_row_as_written(ey_uma):
    expected = refcat2_table()
    table = CatalogStore(DATA / "store").cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg)
    table.sort("RA_ICRS")
    expected.sort("RA_ICRS")
    assert_tables_equal(expected, table)
    # Blanks in the catalog come back masked, not as zeros or NaN.
    assert np.ma.getmaskarray(table["e_pmRA"]).sum() == 2
    assert np.ma.getmaskarray(table["e_Gmag"]).sum() == 2
    assert table.meta["local_store_version"] == "golden test store"
    assert table.meta["local_store"].endswith("refcat2/data-20261117T204712-c532c0dc")


def test_cone_search_honors_radius_and_filters(ey_uma):
    everything = refcat2_table()
    store = CatalogStore(DATA / "store")

    small = store.cone_search(REFCAT2_ID, ey_uma, 0.1 * u.deg)
    sep = ey_uma.separation(SkyCoord(small["RA_ICRS"], small["DE_ICRS"], unit="deg"))
    assert 0 < len(small) < len(everything)
    assert (sep <= 0.1 * u.deg).all()

    limited = store.cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg, {"rmag": "<=14"})
    assert len(limited) == np.sum(np.ma.filled(everything["rmag"] <= 14, False))


def test_radius_takes_any_angle_unit_and_bare_numbers_are_degrees(ey_uma):
    store = CatalogStore(DATA / "store")
    in_deg = store.cone_search(REFCAT2_ID, ey_uma, 0.1 * u.deg)
    in_arcmin = store.cone_search(REFCAT2_ID, ey_uma, 6 * u.arcmin)
    bare = store.cone_search(REFCAT2_ID, ey_uma, 0.1)
    assert len(in_deg) > 0
    assert_tables_equal(in_deg, in_arcmin)
    assert_tables_equal(in_deg, bare)


@pytest.mark.parametrize("radius", [-0.1, np.nan, 0.1 * u.s])
def test_bad_radius_is_rejected(ey_uma, radius):
    store = CatalogStore(DATA / "store")
    with pytest.raises((ValueError, u.UnitsError)):
        store.cone_search(REFCAT2_ID, ey_uma, radius)


def test_center_in_another_frame_is_converted_to_icrs(ey_uma):
    store = CatalogStore(DATA / "store")
    expected = store.cone_search(REFCAT2_ID, ey_uma, 0.1 * u.deg)
    table = store.cone_search(REFCAT2_ID, ey_uma.galactic, 0.1 * u.deg)
    assert_tables_equal(expected, table)


def test_center_must_be_a_single_position(ey_uma):
    store = CatalogStore(DATA / "store")
    two = SkyCoord([ey_uma.ra, ey_uma.ra], [ey_uma.dec, ey_uma.dec])
    with pytest.raises(ValueError, match="single position"):
        store.cone_search(REFCAT2_ID, two, 0.1 * u.deg)


@pytest.mark.parametrize(
    "filters",
    [{"rmag": "<=15"}, {"rmag": "13..15"}, {"gmag": ">14"}, {"pmRA": "!=0"}],
)
def test_filters_match_client_side_filtering(ey_uma, filters):
    store = CatalogStore(DATA / "store")
    unfiltered = store.cone_search(REFCAT2_ID, ey_uma, 0.5 * u.deg)
    expected = _apply_column_filters(unfiltered, filters)
    got = store.cone_search(REFCAT2_ID, ey_uma, 0.5 * u.deg, filters)
    assert 0 < len(got) < len(unfiltered)
    assert_tables_equal(expected, got)


def test_masked_value_never_passes_filter(ey_uma):
    # Some rows have pmRA masked; a filter every real value passes must still
    # drop them.
    store = CatalogStore(DATA / "store")
    everything = store.cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg)
    n_masked = np.ma.getmaskarray(everything["pmRA"]).sum()
    assert n_masked > 0
    kept = store.cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg, {"pmRA": ">-1e9"})
    assert len(kept) == len(everything) - n_masked
    assert not np.ma.getmaskarray(kept["pmRA"]).any()


def test_unknown_filter_column_is_a_value_error(ey_uma):
    store = CatalogStore(DATA / "store")
    # Not CatalogUnavailableError: the query is wrong, not the store.
    with pytest.raises(ValueError, match="Cannot filter on 'nope'") as exc:
        store.cone_search(REFCAT2_ID, ey_uma, 0.5 * u.deg, {"nope": "<1"})
    assert not isinstance(exc.value, CatalogUnavailableError)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ra, dec, radius, expected_names",
    [
        (0.0, 0.0, 0.5, {"wrap-west", "wrap-east"}),  # across RA = 0/360
        (180.0, 0.0, 0.5, {"antimeridian"}),
        (10.0, 89.9, 0.5, {"pole-a", "pole-b"}),  # all RAs near the pole
        (10.0, -89.9, 0.5, {"south-pole"}),
        (45.0, 30.0, 0.05, set()),  # empty cone
    ],
)
def test_cone_geometry(tmp_path, ra, dec, radius, expected_names):
    write_stars(
        stars(
            [
                "wrap-west", "wrap-east", "antimeridian", "pole-a", "pole-b",
                "south-pole", "far",
            ],
            [359.8, 0.2, 180.1, 100.0, 300.0, 200.0, 45.0],
            [0.1, -0.1, 0.2, 89.95, 89.8, -89.95, 31.0],
        ),
        tmp_path,
    )  # fmt: skip
    assert star_names(CatalogStore(tmp_path), ra, dec, radius) == expected_names


def test_zone_height_comes_from_the_manifest(tmp_path):
    # Half-degree zones: dec 10.2 and 10.7 are in different zones (200, 201).
    write_stars(
        stars(["low", "high"], [30.0, 30.0], [10.2, 10.7]), tmp_path, zone_deg=0.5
    )
    assert read_manifest(tmp_path, "stars")["zones"] == [200, 201]
    store = CatalogStore(tmp_path)
    assert star_names(store, 30.0, 10.2, 0.1) == {"low"}
    assert star_names(store, 30.0, 10.7, 0.1) == {"high"}
    assert star_names(store, 30.0, 10.45, 0.5) == {"low", "high"}


# ---------------------------------------------------------------------------
# Stores that do not hold the whole catalog
# ---------------------------------------------------------------------------


def test_shallow_store_warns_when_query_goes_deeper(shallow_root, ey_uma):
    store = CatalogStore(shallow_root)
    with pytest.warns(
        CatalogStoreWarning, match="rmag <= 16.0; fainter sources are missing"
    ):
        store.cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg)
    with pytest.warns(CatalogStoreWarning, match="fainter sources are missing"):
        store.cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg, {"rmag": "<=17"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store.cone_search(REFCAT2_ID, ey_uma, 1.0 * u.deg, {"rmag": "<=14"})


def test_is_complete_only_within_the_stores_depth(shallow_root):
    store = CatalogStore(shallow_root)
    assert store.is_complete(REFCAT2_ID, {"rmag": "<=14"})
    assert store.is_complete(REFCAT2_ID, {"rmag": "<16"})
    assert not store.is_complete(REFCAT2_ID, {"rmag": "<=17"})
    # A range is never counted as an upper bound, even inside the limit.
    assert not store.is_complete(REFCAT2_ID, {"rmag": "10..14"})
    assert not store.is_complete(REFCAT2_ID, {"gmag": "<=14"})
    assert not store.is_complete(REFCAT2_ID)
    # A store without a recorded limit is the whole catalog.
    assert CatalogStore(DATA / "store").is_complete(REFCAT2_ID)


# ---------------------------------------------------------------------------
# Damaged or missing data
# ---------------------------------------------------------------------------

# Three stars in zones 100, 101 and 102 (zone = floor(dec + 90)).
_THREE_ZONES = stars(["z100", "z101", "z102"], [30.0, 30.0, 30.0], [10.5, 11.5, 12.5])


@pytest.mark.parametrize("record_zones", [True, False])
def test_missing_zone_is_unavailable(tmp_path, record_zones):
    # Without the list in the manifest (older stores) the zones come from disk.
    write_stars(_THREE_ZONES, tmp_path, record_zones=record_zones)
    shutil.rmtree(data_dir(tmp_path, "stars") / "zone=101")
    store = CatalogStore(tmp_path)

    with pytest.raises(CatalogUnavailableError, match=r"zone.*101"):
        star_names(store, 30.0, 11.5, 0.2)
    # Zones that are all present still answer...
    assert star_names(store, 30.0, 12.5, 0.2) == {"z102"}
    # ...and so does a cone outside the zones the store was written for.
    assert star_names(store, 30.0, -30.0, 0.2) == set()
    assert star_names(store, 30.0, 60.0, 0.2) == set()


def test_zone_deleted_after_opening_is_unavailable(tmp_path):
    write_stars(_THREE_ZONES, tmp_path)
    store = CatalogStore(tmp_path)
    assert star_names(store, 30.0, 11.5, 0.2) == {"z101"}
    shutil.rmtree(data_dir(tmp_path, "stars") / "zone=101")
    with pytest.raises(CatalogUnavailableError, match=r"zone.*101"):
        star_names(store, 30.0, 11.5, 0.2)


def test_unreadable_data_is_unavailable(golden_root, ey_uma):
    store = CatalogStore(golden_root)
    assert len(store.cone_search(REFCAT2_ID, ey_uma, 0.5 * u.deg)) > 0
    # A file is corrupt...
    data = data_dir(golden_root, "refcat2")
    for parquet in data.rglob("*.parquet"):
        parquet.write_bytes(b"not a parquet file")
    with pytest.raises(CatalogUnavailableError, match="Could not read"):
        store.cone_search(REFCAT2_ID, ey_uma, 0.5 * u.deg)
    # ...or the data goes away altogether (external disk unmounted).
    shutil.rmtree(data)
    with pytest.raises(CatalogUnavailableError, match="Could not read"):
        store.cone_search(REFCAT2_ID, ey_uma, 0.5 * u.deg)


def test_repr_names_root_and_catalogs():
    text = repr(CatalogStore(DATA / "store"))
    assert "CatalogStore" in text
    assert REFCAT2_ID in text
    assert STARS_ID not in text
