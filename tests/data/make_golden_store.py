"""
Regenerate the golden store in this directory.

Run with an environment that has stellarphot >= 2.2.0b4 installed::

    python make_golden_store.py

The store is written by stellarphot's own ``write_local_store``, so the tests
that read it check this package against what the real writer produces rather
than against the simplified writer in ``conftest.py``.

Input is stellarphot's refcat2 test fixture for the field of EY UMa (187
stars from ATLAS-REFCAT2, Tonry et al. 2018). A few values are masked, the
way VizieR blanks them, so that the null round trip is exercised. The masked
table is also saved as ``refcat2_ey_uma.ecsv``: it is what a cone covering
the whole field must return.
"""

import shutil
from pathlib import Path

from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from stellarphot.catalog_backends import write_local_store

HERE = Path(__file__).parent

raw = Table(
    Table.read(
        get_pkg_data_filename(
            "data/all_refcat2_ey_uma.ecsv", package="stellarphot.tests"
        )
    ),
    masked=True,
)
raw.remove_columns(["_r", "id"])
raw["e_pmRA"].mask[:2] = True
raw["e_pmDE"].mask[:2] = True
raw["e_Gmag"].mask[2:4] = True
raw["pmRA"].mask[5] = True

shutil.rmtree(HERE / "store", ignore_errors=True)
write_local_store(
    raw,
    HERE / "store",
    "refcat2",
    "J/ApJ/867/105/refcat2",
    "RA_ICRS",
    "DE_ICRS",
    version="golden test store",
)
raw.write(HERE / "refcat2_ey_uma.ecsv", overwrite=True)
