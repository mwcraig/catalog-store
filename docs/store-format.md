# Store format, version 1

This describes what `catalog-store-reader` reads. The format string in the
manifest is `stellarphot-catalog-store` because stellarphot's
`write_local_store` defined it.

## Layout

```
<root>/
  <name>/                     one directory per catalog, e.g. refcat2/
    catalog.json              the manifest
    <data_dir>/               named by the manifest, e.g. data-20261105T093012-1a2b3c4d/
      zone=0/
        part-0-0.parquet
      zone=1/
        ...
      zone=179/
        part-0-0.parquet
        part-1-0.parquet      a zone may have any number of files
```

The reader finds catalogs by looking for `<root>/*/catalog.json`.

## Declination zones

Rows are partitioned by declination into zones of height `zone_deg` degrees
(1 degree unless the manifest says otherwise):

```
zone = floor((dec + 90) / zone_deg)
```

With 1-degree zones, zone 0 is declination -90 to -89 and zone 179 is +89 to
+90. A zone's directory is `zone=<N>`, with no zero padding. This is Hive
partitioning: the zone number is in the directory name only and is not a
column in the Parquet files. The reader does not return it.

Zones with no rows have no directory.

## Parquet files

- One row per catalog source. Column names are the catalog's own (VizieR's
  names, in the stores built so far).
- All files of a catalog have the same schema.
- A blank value in the catalog is a Parquet null. The reader returns it
  masked.
- Text columns are UTF-8 strings.
- The two coordinate columns named by the manifest are floating-point ICRS
  degrees, with right ascension in the range 0 to 360.
- Within each file, rows are sorted by right ascension and written in small
  row groups (20,000 rows in stellarphot's writer). This is what makes a cone
  search fast: the reader pushes a right-ascension and declination bounding
  box into the scan, and row-group statistics let it skip nearly all of a
  zone. A store without the sorting is still read correctly, only slowly.
- Compression is zstd in stellarphot's writer; the reader accepts any codec
  pyarrow can read.

File names inside a zone directory do not matter.

## Manifest

`catalog.json` is one JSON object.

| Key | Required | Meaning |
| --- | --- | --- |
| `format` | yes | Must be `"stellarphot-catalog-store"`. |
| `format_version` | no | Must be `1` if present; taken as `1` if absent. |
| `vizier_id` | yes | The catalog's identifier, e.g. `"J/ApJ/867/105/refcat2"`. Queries name the catalog by this or by the directory name. |
| `ra_col`, `dec_col` | yes | Names of the right ascension and declination columns. |
| `data_dir` | no | Sub-directory holding the zone directories. Default `"data"`. |
| `zone_deg` | no | Zone height in degrees. Default `1.0`. |
| `units` | no | Object mapping column name to an astropy unit string, e.g. `{"rmag": "mag", "pmRA": "mas / yr"}`. Columns not listed have no unit. |
| `mag_limit` | no | Object mapping column name to a magnitude, or `null`. See below. |
| `zones` | no | Sorted list of the zone numbers that have rows. See below. |
| `version` | no | Free-form provenance string. Returned in the result's `meta` as `local_store_version`. |
| `name`, `layout`, `rows` | no | Informational: directory name, `"dec_zone_parquet"`, and the row count. The reader ignores them. |

The reader skips, with a warning, any catalog whose manifest is unreadable,
has a different `format`, has a `format_version` other than 1, or lacks a
required key.

### `mag_limit`

A store may hold only the bright part of a catalog. `mag_limit` then says how
deep it goes: with `{"gmag": 19.0, "rmag": 19.0, "imag": 19.0}`, every source
at or brighter than 19.0 in at least one of the three columns is present.
Absent or `null` means the store is the whole catalog.

The reader uses this only to warn when a query may reach fainter than the
store, and to answer `is_complete`.

### `zones`

When the manifest lists its zones, the reader treats a listed zone whose
directory is missing as lost data and raises `CatalogUnavailableError` for
any cone that needs it. A zone that is not listed had no rows.

When the list is absent (stores written before it was recorded), the reader
lists the zone directories on disk when the store is opened, and treats any
gap between the lowest and highest zone as lost data.

In both cases a cone outside the lowest-to-highest range is simply outside
the store's sky coverage and returns no rows.

## Replacing a store

stellarphot's writer builds a new `data-<timestamp>-<token>` directory, then
replaces `catalog.json` in one step so that `data_dir` points at it, then
removes the old data directory. A reader opened before the switch keeps the
old `data_dir` and will raise `CatalogUnavailableError` once that directory
is gone; open the store again to pick up the new one.
