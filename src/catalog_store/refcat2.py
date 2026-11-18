"""
Build a catalog store of ATLAS-REFCAT2 (Tonry et al. 2018) from the tile
files MAST distributes.

MAST publishes the catalog in magnitude bins (``00_m_16``, ``16_m_17``, ...),
each a directory of 64,800 one-degree tiles named ``RRR+DD.rc2``. Give this
module the bins, brightest first, and it writes a store complete to the upper
edge of the faintest one::

    build-refcat2-store --tiles 00_m_16 --tiles 16_m_17 --out /srv/catalogs

or ``python -m catalog_store.refcat2`` with the same arguments.
``docs/building-refcat2.md`` in the repository has the download locations,
sizes and timings.

The store has the column names, units, precision and blanks that VizieR
serves for ``J/ApJ/867/105/refcat2``, so code written against VizieR's copy
reads it unchanged.
"""

import argparse
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
from astropy import units as u
from astropy.table import MaskedColumn, Table

from .writer import _table_to_arrow, _write_manifest, _write_parts

__all__ = ["REFCAT2_ID", "build_refcat2_store", "main"]

REFCAT2_ID = "J/ApJ/867/105/refcat2"

_RA_COL, _DEC_COL, _ZONE_DEG = "RA_ICRS", "DE_ICRS", 1.0
_TILES_PER_BIN = 360 * 180
_BIN_RE = re.compile(r"^(\d{2})_m_(\d{2})$")

# One entry per column of a tile, in order:
# (VizieR name, divisor to VizieR's value, unit, dtype, blank when stored 0)
#
# The tiles hold scaled integers (Tonry et al. 2018, Table 2): 1e-8 deg,
# 10 uas, 10 uas/yr, mmag, 0.1 arcsec and 0.01 for the chi values. VizieR
# serves a stored 0 as a blank in most columns. A divisor of None marks the
# "contrib" columns, two hexadecimal digits that are kept as text.
_COLUMNS = [
    ("RA_ICRS", 1e8, u.deg, "f8", False),
    ("DE_ICRS", 1e8, u.deg, "f8", False),
    ("Plx", 100, u.mas, "f8", True),
    ("e_Plx", 100, u.mas, "f4", True),
    ("pmRA", 100, u.mas / u.yr, "f8", True),
    ("e_pmRA", 100, u.mas / u.yr, "f4", True),
    ("pmDE", 100, u.mas / u.yr, "f8", True),
    ("e_pmDE", 100, u.mas / u.yr, "f4", True),
    ("Gmag", 1000, u.mag, "f4", True),
    ("e_Gmag", 1000, u.mag, "f4", True),
    ("BPmag", 1000, u.mag, "f4", True),
    ("e_BPmag", 1000, u.mag, "f8", True),
    ("RPmag", 1000, u.mag, "f4", True),
    ("e_RPmag", 1000, u.mag, "f4", True),
    ("Teff", 1, u.K, "i2", True),
    ("AG", 1000, u.mag, "f4", True),
    ("dupvar", 1, None, "u1", False),
    ("Ag", 1000, u.mag, "f8", False),
    ("rp1", 10, u.arcsec, "f4", False),
    ("r1", 10, u.arcsec, "f4", False),
    ("r10", 10, u.arcsec, "f4", False),
    ("gmag", 1000, u.mag, "f4", True),
    ("e_gmag", 1000, u.mag, "f4", True),
    ("gchi", 100, None, "f4", False),
    ("gcontrib", None, None, "U2", False),
    ("rmag", 1000, u.mag, "f4", True),
    ("e_rmag", 1000, u.mag, "f4", True),
    ("rchi", 100, None, "f4", False),
    ("rcontrib", None, None, "U2", False),
    ("imag", 1000, u.mag, "f4", True),
    ("e_imag", 1000, u.mag, "f4", True),
    ("ichi", 100, None, "f4", False),
    ("icontrib", None, None, "U2", False),
    ("zmag", 1000, u.mag, "f4", True),
    ("e_zmag", 1000, u.mag, "f4", True),
    ("zchi", 100, None, "f4", False),
    ("zcontrib", None, None, "U2", False),
    ("nstat", 1, None, "i2", False),
    ("Jmag", 1000, u.mag, "f4", True),
    ("e_Jmag", 1000, u.mag, "f4", True),
    ("Hmag", 1000, u.mag, "f4", True),
    ("e_Hmag", 1000, u.mag, "f4", True),
    ("Kmag", 1000, u.mag, "f4", True),
    ("e_Kmag", 1000, u.mag, "f4", True),
]
_NAMES = [column[0] for column in _COLUMNS]
_UNITS = {name: unit.to_string() for name, _, unit, _, _ in _COLUMNS if unit}
_READ_OPTIONS = pacsv.ReadOptions(column_names=_NAMES, use_threads=False)
_CONVERT_OPTIONS = pacsv.ConvertOptions(
    column_types={
        name: pa.string() if divisor is None else pa.int64()
        for name, divisor, _, _, _ in _COLUMNS
    }
)


def build_refcat2_store(
    tile_dirs,
    root,
    name="refcat2",
    mag_limit=None,
    dec_min=-90,
    dec_max=89,
    threads=8,
    resume=False,
    require_all_tiles=True,
    progress=print,
):
    """
    Build a store of ATLAS-REFCAT2 from MAST's tile files.

    Parameters
    ----------
    tile_dirs : list of str or `pathlib.Path`
        One directory per MAST magnitude bin, each named as MAST names it
        (``00_m_16``, ``16_m_17``, ...) and holding that bin's ``.rc2``
        tiles. The bins must start at ``00_m_16`` and be contiguous.
    root : str or `pathlib.Path`
        Store directory; the catalog goes in ``root / name``.
    name : str, optional
        Directory name for the catalog.
    mag_limit : float, optional
        Depth to record for the store. Defaults to the upper edge of the
        faintest bin, and cannot be fainter than that.
    dec_min, dec_max : int, optional
        First and last declination band to build, named by its lower edge
        in degrees: -90 to 89 is the whole sky.
    threads : int, optional
        Number of tiles read at once.
    resume : bool, optional
        Continue a build that was interrupted instead of starting again.
    require_all_tiles : bool, optional
        Refuse to build unless every bin has all 64,800 tiles. MAST publishes
        no checksums, so this is the check that a download and unpacking
        were complete. Set to ``False`` to build from part of the sky.
    progress : callable, optional
        Called with a line of text after each (declination, bin) is written.
        ``None`` for silence.

    Returns
    -------
    `pathlib.Path`
        Path to the catalog directory.

    Raises
    ------
    ValueError
        If the bins cannot give a store complete to ``mag_limit``, a
        directory is not named like a bin, or tiles are missing and
        ``require_all_tiles`` is set. Nothing is changed on disk.

    Notes
    -----
    A bin holds the stars whose brightest of g, r and i is in its range, so
    the store is complete to ``mag_limit`` in the sense that every star at or
    brighter than the limit in at least one of g, r and i is present. That is
    what the manifest's ``mag_limit`` records for the three columns.

    Each (declination band, bin) pair becomes one Parquet file, so peak
    memory is one bin's stars in one band: a few million rows at most.

    A build that is not a resume first removes the catalog's manifest and
    data, so the catalog is absent from the store until the build finishes;
    the manifest is written last. With ``resume``, a band that already has
    one file per bin is kept and any other band found on disk is rebuilt.
    """
    progress = progress or (lambda message: None)
    tile_dirs = sorted((Path(d) for d in tile_dirs), key=_bin_edges)
    edges = [_bin_edges(d) for d in tile_dirs]
    bins = ", ".join(d.name for d in tile_dirs)
    if edges[0][0] != 0:
        raise ValueError(
            f"bins must start at 00_m_16 for the store to be complete; got {bins}"
        )
    for (_, upper), (lower, _) in zip(edges[:-1], edges[1:], strict=True):
        if upper != lower:
            raise ValueError(f"bins are not contiguous: {bins}")
    faintest = float(edges[-1][1])
    mag_limit = faintest if mag_limit is None else float(mag_limit)
    if mag_limit > faintest:
        raise ValueError(f"mag_limit {mag_limit:g} exceeds the faintest bin, {bins}")
    if require_all_tiles:
        for tile_dir in tile_dirs:
            found = sum(1 for _ in tile_dir.glob("*.rc2"))
            if found != _TILES_PER_BIN:
                raise ValueError(
                    f"{tile_dir}: expected {_TILES_PER_BIN} tiles, found {found}"
                )

    cat_dir = Path(root) / name
    data_dir = cat_dir / "data"
    if not resume:
        (cat_dir / "catalog.json").unlink(missing_ok=True)
        shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(threads) as pool:
        for dec in range(dec_min, dec_max + 1):
            zone = int(np.floor((dec + 90) / _ZONE_DEG))
            zone_dir = data_dir / f"zone={zone}"
            if zone_dir.exists():
                have = len(list(zone_dir.glob("*.parquet")))
                if resume and have >= len(tile_dirs):
                    progress(f"dec {dec:+03d}: kept, {have} files present")
                    continue
                shutil.rmtree(zone_dir)
            for tile_dir in tile_dirs:
                start = time.perf_counter()
                table = _read_band(tile_dir, dec, pool, require_all_tiles)
                if table is None:
                    continue
                arrow_table = _table_to_arrow(table, _RA_COL, _DEC_COL, _ZONE_DEG, {})
                _write_parts(arrow_table, data_dir, f"part-z{zone:03d}-{tile_dir.name}")
                progress(
                    f"dec {dec:+03d} {tile_dir.name}: {len(table):>9d} rows "
                    f"({time.perf_counter() - start:.1f} s)"
                )

    zones = [
        int(zone_dir.name.partition("=")[2])
        for zone_dir in data_dir.glob("zone=*")
        if any(zone_dir.glob("*.parquet"))
    ]
    rows = ds.dataset(data_dir, format="parquet", partitioning="hive").count_rows()
    files = ", ".join(
        f"hlsp_atlas-refcat2_atlas_ccd_{lower:02d}-m-{upper:02d}_multi_v1_cat"
        for lower, upper in edges
    )
    _write_manifest(
        cat_dir,
        catalog_id=REFCAT2_ID,
        name=name,
        zone_deg=_ZONE_DEG,
        ra_col=_RA_COL,
        dec_col=_DEC_COL,
        data_dir=data_dir.name,
        rows=rows,
        version=(
            f"ATLAS refcat2 (Tonry et al. 2018) m<{mag_limit:g} from bins {bins}, "
            f"MAST HLSP {files}; built {time.strftime('%Y-%m-%d')}"
        ),
        mag_limit={"gmag": mag_limit, "rmag": mag_limit, "imag": mag_limit},
        units=_UNITS,
        zones=zones,
    )
    return cat_dir


def _bin_edges(tile_dir):
    """
    Return the magnitude range of a MAST bin from its directory name.

    Parameters
    ----------
    tile_dir : `pathlib.Path`
        A bin directory, e.g. ``.../16_m_17``.

    Returns
    -------
    tuple of int
        Lower and upper magnitude edges, e.g. ``(16, 17)``.

    Raises
    ------
    ValueError
        If the directory is not named like a bin.
    """
    match = _BIN_RE.match(Path(tile_dir).name)
    if match is None:
        raise ValueError(
            f"{tile_dir}: directory name is not a MAST refcat2 bin like 16_m_17"
        )
    return int(match.group(1)), int(match.group(2))


def _read_tile(path):
    """
    Read one tile.

    Parameters
    ----------
    path : `pathlib.Path`
        A ``.rc2`` file: comma-separated, no header, 44 columns.

    Returns
    -------
    `pyarrow.Table` or None
        The tile's rows as stored (scaled integers, text for the contrib
        columns), or `None` for a tile with no stars, which is an empty file.
    """
    if path.stat().st_size == 0:
        return None
    return pacsv.read_csv(
        path, read_options=_READ_OPTIONS, convert_options=_CONVERT_OPTIONS
    )


def _read_band(tile_dir, dec, pool, require_all_tiles):
    """
    Read the 360 tiles of one declination band of one bin.

    Parameters
    ----------
    tile_dir : `pathlib.Path`
        The bin's directory.
    dec : int
        Lower edge of the band in degrees.
    pool : `concurrent.futures.Executor`
        Where the tiles are read.
    require_all_tiles : bool
        If false, tiles that do not exist are skipped.

    Returns
    -------
    `astropy.table.Table` or None
        The band's stars with VizieR's columns, or `None` if it has none.
    """
    paths = [tile_dir / f"{ra:03d}{dec:+03d}.rc2" for ra in range(360)]
    if not require_all_tiles:
        paths = [path for path in paths if path.exists()]
    tiles = [tile for tile in pool.map(_read_tile, paths) if tile is not None]
    if not tiles:
        return None
    return _tiles_to_table(pa.concat_tables(tiles))


def _tiles_to_table(raw):
    """
    Convert rows as stored in the tiles to what VizieR serves.

    Parameters
    ----------
    raw : `pyarrow.Table`
        Rows from `_read_tile`.

    Returns
    -------
    `astropy.table.Table`
        One column per tile column, with VizieR's name, unit and dtype. A
        value VizieR leaves blank is masked.
    """
    table = Table()
    for name, divisor, unit, dtype, blank_when_zero in _COLUMNS:
        if divisor is None:
            padded = pc.utf8_lpad(raw[name], width=2, padding="0")
            table[name] = padded.to_numpy(zero_copy_only=False).astype(dtype)
            continue
        stored = raw[name].to_numpy()
        values = (
            stored.astype(dtype) if divisor == 1 else (stored / divisor).astype(dtype)
        )
        mask = (stored == 0) if blank_when_zero else np.zeros(len(stored), dtype=bool)
        table[name] = MaskedColumn(values, mask=mask, unit=unit)
    return table


def main(argv=None):
    """
    Run the builder from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; defaults to the process's.
    """
    parser = argparse.ArgumentParser(
        prog="build-refcat2-store",
        description="Build a catalog store of ATLAS-REFCAT2 from MAST's tile files.",
    )
    parser.add_argument(
        "--tiles",
        action="append",
        required=True,
        metavar="DIR",
        help="a MAST bin directory such as .../00_m_16; repeat for each bin",
    )
    parser.add_argument("--out", required=True, help="store directory")
    parser.add_argument("--name", default="refcat2", help="catalog directory name")
    parser.add_argument(
        "--mag-limit",
        type=float,
        default=None,
        help="depth to record (default: upper edge of the faintest bin)",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dec-min", type=int, default=-90)
    parser.add_argument("--dec-max", type=int, default=89)
    parser.add_argument(
        "--resume", action="store_true", help="continue an interrupted build"
    )
    parser.add_argument(
        "--allow-missing-tiles",
        action="store_true",
        help="build even if a bin does not have all 64,800 tiles",
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    try:
        cat_dir = build_refcat2_store(
            args.tiles,
            args.out,
            name=args.name,
            mag_limit=args.mag_limit,
            dec_min=args.dec_min,
            dec_max=args.dec_max,
            threads=args.threads,
            resume=args.resume,
            require_all_tiles=not args.allow_missing_tiles,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except ValueError as err:
        raise SystemExit(f"build-refcat2-store: {err}") from err
    rows = ds.dataset(cat_dir / "data", format="parquet").count_rows()
    print(f"wrote {cat_dir}: {rows} rows in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
