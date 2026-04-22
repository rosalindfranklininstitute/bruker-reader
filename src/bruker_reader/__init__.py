from pathlib import Path
import os
import sys
from contextlib import contextmanager

from icecream import ic

import numpy as np


@contextmanager
def load_opentims(path):

    from ctypes import util

    tmp = util.find_library

    def intercept_find(name: str) -> str | None:
        if name == "sqlite3":
            return str(Path(os.path.dirname(__file__)) / "dlls" / "sqlite3.dll")
        else:
            return tmp(name)

    util.find_library = intercept_find

    try:
        from opentimspy import OpenTIMS
        import opentimspy

        if not opentimspy.bruker_bridge_present:
            opentimspy.setup_opensource()

        path = Path(
            "C:/Workspace/data/2024_06_19_MBrain_ABC_DP2/2024_06_19_MBrain_ABC_DP2.d/"
        )
        with OpenTIMS(path) as D:
            yield D

    finally:
        util.find_library = tmp


def main() -> None:

    path = Path(
        "C:/Workspace/data/2024_06_19_MBrain_ABC_DP2/2024_06_19_MBrain_ABC_DP2.d/"
    )
    all_columns = (
        "frame",
        "scan",
        "tof",
        "intensity",
        "mz",
        "inv_ion_mobility",
        "retention_time",
    )
    with load_opentims(path) as D:
        ic(D)
        ic(D.all_columns)
        ic(D.frames)
        ic(D.frames["NumPeaks"])
        ic(D.frames["NumScans"])
        ic(np.sum(D.frames["NumPeaks"]))
        ic(np.sum(D.frames["NumScans"]))
        # for dd in sorted(dir(D)):
        #     print(f"{dd: <40}: {type(getattr(D, dd))}")
        ic(D.min_frame, D.max_frame)
        ic(D.min_scan, D.max_scan)
        ic(D.min_mz, D.max_mz)
        ic(D.min_inv_ion_mobility, D.max_inv_ion_mobility)
