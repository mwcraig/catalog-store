# catalog-store-reader

Cone searches on an offline copy of an astronomical catalog.

A *store* is a directory of Parquet files, partitioned by declination, with a
small JSON manifest per catalog. This package reads one: give it a position
and a radius and it returns the catalog rows in that cone as an astropy
`Table`. No network, no database server. A cone search on a 785-million-row
copy of ATLAS-REFCAT2 takes a few hundredths of a second.

The package also writes stores: `write_store` for any catalog you have as a
table, and a `build-refcat2-store` command that builds ATLAS-REFCAT2 from the
files MAST distributes. See [Writing a store](#writing-a-store) below.

The on-disk layout is described in
[docs/store-format.md](docs/store-format.md), so a store can also be written
or read by other tools. The code was extracted from
[stellarphot](https://github.com/feder-observatory/stellarphot), whose own
stores are in the same format.

## Install

```
pip install .
```

Requires Python 3.11 or later, numpy, pyarrow and astropy. The reader is the
single file `src/catalog_store_reader/reader.py`, which can also be copied
into another project as is.

## Reading a store

```python
from astropy import units as u
from astropy.coordinates import SkyCoord

from catalog_store_reader import CatalogStore

store = CatalogStore("/path/to/store")
store.catalog_ids  # ('J/ApJ/867/105/refcat2',)

stars = store.cone_search(
    "refcat2",  # or "J/ApJ/867/105/refcat2"
    SkyCoord(135.55, 49.84, unit="deg"),  # any frame; one position
    10 * u.arcmin,  # a bare number means degrees
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

## Writing a store

### ATLAS-REFCAT2

```
build-refcat2-store --tiles /data/00_m_16 --tiles /data/16_m_17 --out /srv/catalogs
```

builds `/srv/catalogs/refcat2` from the tile files MAST distributes, with the
column names, units and blanks VizieR serves for `J/ApJ/867/105/refcat2`.
[docs/building-refcat2.md](docs/building-refcat2.md) covers where to download
the tiles, how much disk each depth needs and how long a build takes.

### Any other catalog

```python
from catalog_store_reader import write_store

write_store(
    table,  # an astropy Table, or an iterable of Tables
    "/srv/catalogs",
    "mycat",  # directory name
    "J/A+A/000/00/mycat",  # identifier queries will use
    "RAJ2000",  # coordinate columns, ICRS degrees
    "DEJ2000",
    version="where this copy came from",
)
```

Masked values are stored as blanks and come back masked. A catalog too large
for memory can be passed as an iterable of tables. Writing over an existing
catalog builds the new copy alongside it and switches in one step, so an
interrupted write leaves the old copy in place.

## Tests

```
uv run --group test pytest
```

`tests/data/store` is a 187-star store written by stellarphot's writer, so
the reader is tested against a store this package did not write;
`tests/data/make_golden_store.py` regenerates it. `tests/data/tiles` holds
excerpts of eight real REFCAT2 tiles covering the same field; the builder
must turn them into exactly the values VizieR returned for it.

## License

BSD 3-clause, the same as stellarphot; see [LICENSE](LICENSE).
