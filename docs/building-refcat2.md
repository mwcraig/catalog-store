# Building an ATLAS-REFCAT2 store

ATLAS-REFCAT2 (Tonry et al. 2018, ApJ 867, 105) is an all-sky catalog of
about a billion stars to roughly magnitude 19, with Gaia DR2 astrometry and
g, r, i, z photometry on the Pan-STARRS system. MAST distributes it as a
High Level Science Product: <https://archive.stsci.edu/hlsp/atlas-refcat2>
(DOI [10.17909/t9-2p3r-7651](http://dx.doi.org/10.17909/t9-2p3r-7651)).

`build-refcat2-store` turns MAST's files into a store this package reads.

## 1. Decide how deep to go

MAST splits the catalog into bins by *m*, the brightest of a star's g, r and
i magnitudes. A store to a given depth needs every bin from the brightest up
to that depth.

| Depth | Bins needed | Download | Unpacked tiles | Stars | Store size |
| --- | --- | --- | --- | --- | --- |
| m < 16 | `00-m-16` | 6.3 GB | 19 GB | 105.5 M | 7.2 GB |
| m < 17 | + `16-m-17` | 12.3 GB | about 37 GB | about 212 M | not built |
| m < 18 | + `17-m-18` | 22.8 GB | about 68 GB | about 416 M | not built |
| m < 19 | + `18-m-19` | 40.1 GB | 128 GB | 785.5 M | 43 GB |

The m < 16 and m < 19 rows are measured from stores that were built. The
other two are estimates from the download sizes and the star counts in the
paper.

There is also a `19-m-20` bin (9.3 GB). The catalog is very incomplete
there, and no store has been built with it.

Because a bin is defined by the *brightest* of g, r and i, a store to m < 19
has every star at or brighter than 19 in at least one of the three bands. A
red star with r = 19.5 and i = 18.5 is in it; so the store is not cut at
r < 19, and it is complete in r only to somewhat brighter than 19.

## 2. Download and unpack

The files are at `https://archive.stsci.edu/hlsps/atlas-refcat2/orig/`:

```
hlsp_atlas-refcat2_atlas_ccd_00-m-16_multi_v1_cat.tbz
hlsp_atlas-refcat2_atlas_ccd_16-m-17_multi_v1_cat.tbz
hlsp_atlas-refcat2_atlas_ccd_17-m-18_multi_v1_cat.tbz
hlsp_atlas-refcat2_atlas_ccd_18-m-19_multi_v1_cat.tbz
```

For example:

```
curl -O https://archive.stsci.edu/hlsps/atlas-refcat2/orig/hlsp_atlas-refcat2_atlas_ccd_00-m-16_multi_v1_cat.tbz
tar -xjf hlsp_atlas-refcat2_atlas_ccd_00-m-16_multi_v1_cat.tbz
```

Each archive unpacks to a directory named for its bin (`00_m_16`, `16_m_17`,
...) holding 64,800 files, one per square degree, named `RRR+DD.rc2` for the
tile covering right ascension RRR to RRR+1 and declination DD to DD+1. Keep
the directory names: the builder reads the magnitude range from them.

MAST publishes no checksums. The builder instead refuses to start unless
every bin has all 64,800 tiles, which catches an interrupted download or
unpacking.

## 3. Build

```
build-refcat2-store \
    --tiles /data/refcat2/00_m_16 \
    --tiles /data/refcat2/16_m_17 \
    --tiles /data/refcat2/17_m_18 \
    --tiles /data/refcat2/18_m_19 \
    --out /srv/catalogs
```

This writes `/srv/catalogs/refcat2`. Progress goes to standard error, one
line per declination band and bin.

| Option | Meaning |
| --- | --- |
| `--tiles DIR` | A bin directory. Repeat for each bin; order does not matter. |
| `--out DIR` | The store directory. |
| `--name NAME` | Catalog directory name. Default `refcat2`. |
| `--mag-limit M` | Depth to record in the manifest. Default: the upper edge of the faintest bin. |
| `--resume` | Continue an interrupted build. Bands that are complete are kept. |
| `--threads N` | Tiles read at once. Default 8. |
| `--dec-min D`, `--dec-max D` | Build only some declination bands, named by their lower edge. Default -90 to 89, the whole sky. |
| `--allow-missing-tiles` | Skip the 64,800-tiles check, to build from part of the sky. |

Building the band at declination +49 to m < 19 (3.1 million stars) took 12
seconds on an Apple-silicon Mac with the tiles on external drives.
Extrapolating from that one band, the whole sky to m < 19 is about an hour;
a full build with this command has not been timed. Peak memory is one bin's
stars in one band, a few million rows.

The catalog does not appear in the store until the build finishes: the
manifest is written last. Starting a build without `--resume` removes any
existing copy of the catalog first.

The same build from Python:

```python
from catalog_store_reader.refcat2 import build_refcat2_store

build_refcat2_store(
    ["/data/refcat2/00_m_16", "/data/refcat2/16_m_17"],
    "/srv/catalogs",
)
```

## 4. Check it

```python
from astropy import units as u
from astropy.coordinates import SkyCoord

from catalog_store_reader import CatalogStore

store = CatalogStore("/srv/catalogs")
print(store.manifest("refcat2")["rows"])
stars = store.cone_search(
    "refcat2", SkyCoord(135.55, 49.84, unit="deg"), 0.25 * u.deg, {"rmag": "<=15"}
)
print(len(stars))
```

A store to m < 16 or deeper returns 58 stars for that query.

## What is in the store

The tiles hold scaled integers. The builder converts them to the values,
column names, units and floating-point types that VizieR serves for
`J/ApJ/867/105/refcat2`, so the store can stand in for VizieR's copy:

| Columns | Stored in the tiles as | In the store |
| --- | --- | --- |
| `RA_ICRS`, `DE_ICRS` | 1e-8 degree | degrees (Gaia DR2, epoch 2015.5) |
| `Plx`, `e_Plx` | 10 microarcsec | mas |
| `pmRA`, `e_pmRA`, `pmDE`, `e_pmDE` | 10 microarcsec/yr | mas/yr |
| magnitudes and their errors, `AG`, `Ag` | millimagnitudes | mag |
| `rp1`, `r1`, `r10` | 0.1 arcsec | arcsec |
| `gchi`, `rchi`, `ichi`, `zchi` | 0.01 | unitless |
| `Teff`, `dupvar`, `nstat` | as is | K, flags, count |
| `gcontrib`, `rcontrib`, `icontrib`, `zcontrib` | two hexadecimal digits | the same, as text |

Where VizieR serves a blank, the store has a blank: a stored zero in the
parallax, proper-motion, magnitude, magnitude-error, `Teff` and `AG`
columns. `Ag`, the radii, the chi values, `dupvar` and `nstat` keep their
zeros.

Two columns VizieR adds are not in the store: `_r`, the distance from the
center of a query, and `GaiaDR2`, a constant link label.

The test suite checks the conversion against a table VizieR returned: every
star VizieR listed in the test field with m < 19 must come out of the
builder with identical values, types, units and blanks in every column.
