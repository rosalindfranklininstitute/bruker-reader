from ms_nexus_tools.lib.sparse_sampling import SparseSampling
from pathlib import Path
import os
import sys
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm

from icecream import ic

from ms_nexus_tools.api import data_convert

from . import maldi
from . import query
from .loader import LoadOpenTims
from .tsf import TsfDataSource


def query_data() -> None:
    partial_args = query.ProcessArgs.parse_config("query")
    process_args = query.ProcessArgs.parse_interactive(
        "query",
        exclude=["config"],
        args=partial_args.remaining_args,
    )
    query.process(process_args, partial_args.config)


def main() -> None:
    partial_args = maldi.ProcessArgs.parse_config("maldi")
    process_args = maldi.ProcessArgs.parse_interactive(
        "maldi", args=partial_args.remaining_args, exclude=["config"]
    )
    maldi.process(process_args, partial_args.config)


def tof() -> None:
    partial_args = data_convert.ProcessArgs.parse_config("tof")
    process_args = data_convert.ProcessArgs.parse_interactive(
        "tof", args=partial_args.remaining_args, exclude=["config"]
    )
    sampling = SparseSampling(
        downsample_count=10,
        area_positions=np.array([50, 75, 100]),
        area_volumes=np.array([75, 20, 5]),
    )
    process_args.data_source = TsfDataSource(process_args.in_path, sampling=sampling)
    data_convert.process(process_args, partial_args.config)


def tims_tof() -> None:
    partial_args = data_convert.ProcessArgs.parse_config("tims_tof")
    process_args = data_convert.ProcessArgs.parse_interactive(
        "tims_tof", args=partial_args.remaining_args, exclude=["config"]
    )
    sampling = SparseAxisSampling(
        downsample_count=10,
        area_positions=np.array([50, 75, 100]),
        area_volumes=np.array([75, 20, 5]),
    )
    process_args.data_source = TsfDataSource(process_args.in_path, sampling=sampling)
    data_convert.process(process_args, partial_args.config)


def scratch() -> None:

    path = Path(
        "C:/Workspace/data/2024_06_19_MBrain_ABC_DP2/2024_06_19_MBrain_ABC_DP2.d/"
    )
    ic(path)
    with LoadOpenTims(path) as D:
        ic(D)
        ic(D.all_columns)
        ic(D.frames)
        ic(D.frames["NumPeaks"])
        ic(D.frames["NumScans"])
        d = D.table2keyed_dict("MaldiFrameInfo")
        ic(d[1])
        d = D.table2dict("MaldiFrameInfo")
        ic([k for k in d.keys()])

    path = Path("C:/Workspace/data/251119_amylase_highres.d - Copy/")
    ic(path)
    with LoadOpenTims(path) as D:
        ic(D)
        ic(D.all_columns)
        ic(D.frames)
        ic(D.frames["NumPeaks"])
        ic(D.frames["NumScans"])
        d = D.table2keyed_dict("MaldiFrameInfo")
        ic(d[1])
        d = D.table2dict("MaldiFrameInfo")
        ic([k for k in d.keys()])
