# catalog-store-reader

Cone searches on an offline copy of an astronomical catalog.

A *store* is a directory of Parquet files, partitioned by declination, with a
small JSON manifest per catalog. This package reads one: give it a position
and a radius and it returns the catalog rows in that cone as an astropy
`Table`. No network, no database server. A cone search on a 785-million-row
copy of ATLAS-REFCAT2 takes a few hundredths of a second.

This package only reads. Stores are written by
[stellarphot](https://github.com/feder-observatory/stellarphot)
(`stellarphot.catalog_backends.write_local_store`), which is where this code
was extracted from. The on-disk layout is described in
[docs/store-format.md](docs/store-format.md), so a store can also be written
or read by other tools.

## Install

```
pip install .
```

Requires Python 3.11 or later, numpy, pyarrow and astropy. The reader is the
single file `src/catalog_store_reader/reader.py`, which can also be copied
into another project as is.

## Use

```python
from astropy import units as u
from astropy.coordinates import SkyCoord

from catalog_store_reader import CatalogStore

store = CatalogStore("/path/to/store")
store.catalog_ids                # ('J/ApJ/867/105/refcat2',)

stars = store.cone_search(
    "refcat2",                              # or "J/ApJ/867/105/refcat2"
    SkyCoord(135.55, 49.84, unit="deg"),    # any frame; one position
    10 * u.arcmin,                          # a bare number means degrees
    column_filters={"rmag": "<=16"},
)
```

A catalog is named either by its identifier in the manifest (the VizieR
identifier, for the stores built so far) or by the name of its directory.

The result has the catalog's own column names and units. Every column is
masked, and a value the catalog leaves blank is masked. Rows come back in no
particular order. Positions are compared as stored: there is no
proper-motion correction.

### Filters

`column_filters` maps a column name to a limit, in a subset of VizieR's
syntax:

| Filter | Keeps rows where the column is |
| --- | --- |
| `"<=16"`, `"<16"`, `">=3"`, `">3"` | on that side of the number |
| `"=2"` or `"2"` | equal to the number |
| `"!=0"` | not equal to the number |
| `"10..12"` | between the two numbers, inclusive |

A row must pass every filter, and a blank value never passes one.

### Stores that are not the whole catalog

A store may hold only the bright part of a catalog. Its manifest then records
a `mag_limit`, for example `{"gmag": 19.0, "rmag": 19.0, "imag": 19.0}`: every
source at or brighter than the limit in at least one of those columns is
present.

`cone_search` answers from such a store either way, but issues a
`CatalogStoreWarning` unless the query has an upper-bound filter (`<` or
`<=`) at or brighter than the limit on one of those columns. Use
`store.is_complete(catalog, column_filters)` to ask beforehand.

### Errors

| Exception | Meaning |
| --- | --- |
| `CatalogNotFoundError` | The store has no catalog by that identifier or name. |
| `CatalogUnavailableError` | The catalog is there but its data could not be read: disk not mounted, corrupt file, or a declination zone missing. |
| `ValueError` | The query is wrong: malformed filter, unknown column, bad radius, or more than one center. |

A sub-directory whose manifest cannot be used is skipped with a
`CatalogStoreWarning` when the store is opened; the other catalogs still
work.

## Tests

```
uv run --group test pytest
```

`tests/data/store` is a 187-star store written by stellarphot's real writer;
`tests/data/make_golden_store.py` regenerates it. The other test stores are
built on the fly by a small writer in `tests/conftest.py` that follows the
format document.

## License

BSD 3-clause, the same as stellarphot; see [LICENSE](LICENSE).
