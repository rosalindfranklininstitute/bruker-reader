from pathlib import Path
import os
import sys
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm

from icecream import ic

from . import maldi
from .loader import load_opentims


def main() -> None:
    partial_args = maldi.ProcessArgs.parse_config("maldi")
    process_args = maldi.ProcessArgs.parse_interactive(
        "maldi", args=partial_args.remaining_args, exclude=["config"]
    )
    maldi.process(process_args, partial_args.config)


def scratch() -> None:

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
        for dd in sorted(dir(D)):
            print(f"{dd: <40}: {type(getattr(D, dd))}")
        exit()
        ic(D.min_frame, D.max_frame)
        ic(D.min_scan, D.max_scan)
        ic(D.min_mz, D.max_mz)
        d = D.table2keyed_dict("MaldiFrameInfo")
        ic(d[1])
        d = D.table2dict("MaldiFrameInfo")
        ic([k for k in d.keys()])
        x = d["XIndexPos"]
        y = d["YIndexPos"]
        xy = np.array([[x, y] for x, y in zip(x, y)])
        ic(np.unique(xy, axis=0).shape)
        ic((np.min(x), np.max(x)))
        ic((np.min(y), np.max(y)))
        xs = ic(len(np.unique(x)))
        ys = ic(len(np.unique(y)))
        ic(len(xy), xy.shape)
        ic(xs * ys)
        ic(D.min_inv_ion_mobility, D.max_inv_ion_mobility)
        ic(D.frame_properties[1])

        min_tof, max_tof = D.mz_to_tof([D.min_mz, D.max_mz], [1, 1])
        tof_range = max_tof - min_tof
        tof_count = int(tof_range / 100)
        ic(tof_range, tof_count)
        tof_edges = [ii * 100 + min_tof for ii in range(tof_count + 1)]
        ic(len(tof_edges))

        second = False
        unique_tof = np.zeros((tof_count,))
        for frame in tqdm(D.query_iter(D.ms1_frames, columns=all_columns)):
            tof, _ = np.histogram(frame["tof"], bins=tof_edges)

            # ic(frame)
            # ic(np.unique(frame["frame"]))
            # ic(np.unique(frame["retention_time"]))
            # ic(np.unique(frame["scan"]))
            # for dd in sorted(dir(frame)):
            #     print(f"{dd: <40}: {type(getattr(frame, dd))}")
            # if second:
            #     break
            # else:
            #     second = True
            unique_tof += tof
        ic(unique_tof.shape)
        ic(np.max(unique_tof))
        ic(np.min(unique_tof))
        ic(np.median(unique_tof))
