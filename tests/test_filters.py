import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from astropy import units as u
from astropy.table import MaskedColumn, Table

from catalog_store_reader.reader import (
    _apply_column_filters,
    _arrow_to_masked_table,
    _filter_expression,
    _parse_filter,
)


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("<=14", ("<=", 14.0, None)),
        ("<14", ("<", 14.0, None)),
        (">=3", (">=", 3.0, None)),
        (" > 3.5 ", (">", 3.5, None)),
        ("=2", ("=", 2.0, None)),
        ("2", ("=", 2.0, None)),
        ("!=2", ("!=", 2.0, None)),
        ("10..12", ("..", 10.0, 12.0)),
    ],
)
def test_parse_filter(spec, expected):
    assert _parse_filter(spec) == expected


def test_parse_filter_rejects_garbage():
    with pytest.raises(ValueError, match="Unsupported column filter"):
        _parse_filter("<= 14 mag")


def test_apply_column_filters_masks_never_pass():
    table = Table(
        {
            "rmag": MaskedColumn(
                [10.0, 15.0, 20.0, 12.0], mask=[False, False, False, True]
            )
        }
    )
    kept = _apply_column_filters(table, {"rmag": "<=15"})
    assert list(kept["rmag"]) == [10.0, 15.0]
    kept = _apply_column_filters(table, {"rmag": "11..16"})
    assert list(kept["rmag"]) == [15.0]
    assert len(_apply_column_filters(table, {})) == 4
    with pytest.raises(ValueError, match="not in catalog"):
        _apply_column_filters(table, {"nope": "<1"})


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("<=15.1", [14.0, 15.1]),
        ("<15.1", [14.0]),
        (">=15.1", [15.1, 16.0]),
        (">15.1", [16.0]),
        ("=15.1", [15.1]),
        ("!=15.1", [14.0, 16.0]),
        ("15.1..16", [15.1, 16.0]),
        ("14..15.1", [14.0, 15.1]),
    ],
)
def test_filter_expression_float32_limits_and_nulls(spec, expected):
    # 15.1 as float32 is not 15.1 as float64; a star exactly at the limit
    # must still pass, and a null (masked) value must never pass.
    mags = pa.array([14.0, 15.1, 16.0, None], type=pa.float32())
    arrow_table = pa.table({"rmag": mags})
    expr = _filter_expression(pc.field("rmag"), spec, mags.type)
    kept = arrow_table.filter(expr)
    np.testing.assert_array_equal(
        kept["rmag"].to_numpy(), np.array(expected, dtype=np.float32)
    )
    # Same answer as the client-side filter that remains the authority.
    reference = _apply_column_filters(
        _arrow_to_masked_table(arrow_table, {}), {"rmag": spec}
    )
    np.testing.assert_array_equal(kept["rmag"].to_numpy(), reference["rmag"])


def test_filter_expression_integer_and_string_columns():
    flags = pa.array([14, 15, 16, None], type=pa.int16())
    expr = _filter_expression(pc.field("flag"), "<=15.5", flags.type)
    kept = pa.table({"flag": flags}).filter(expr)
    assert kept["flag"].to_pylist() == [14, 15]
    # A string column has no numeric comparison; leave it to the post-filter.
    assert _filter_expression(pc.field("name"), "<=1", pa.string()) is None


def test_arrow_to_masked_table_keeps_integer_dtypes():
    big = 2**60 + 1  # not representable as float64
    arrow_table = pa.table(
        {
            "id": pa.array([big, None, 3], type=pa.int64()),
            "flag": pa.array([1, None, 0], type=pa.uint8()),
            "ok": pa.array([True, None, False], type=pa.bool_()),
            "mag": pa.array([1.5, None, 2.5], type=pa.float32()),
            "name": pa.array(["a", None, "c"], type=pa.string()),
        }
    )
    table = _arrow_to_masked_table(arrow_table, {"mag": "mag"})
    assert table["id"].dtype == np.int64
    assert table["id"][0] == big
    assert table["flag"].dtype == np.uint8
    assert table["ok"].dtype == np.bool_
    assert table["mag"].dtype == np.float32
    assert table["mag"].unit == u.mag
    for colname in table.colnames:
        assert list(np.ma.getmaskarray(table[colname])) == [False, True, False]
