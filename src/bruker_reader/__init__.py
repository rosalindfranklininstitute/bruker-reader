# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from argsui import FilePathType, arg_field
from xartus.api import data_convert
from xartus.lib.multi_linspace import MultiLinspace

from .tdf import TdfDataSource
from .tsf import TsfDataSource

logging.basicConfig(format="%(module)s: %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger(data_convert.__name__).setLevel(logging.INFO)


@dataclass
class BrukerArgs(data_convert.ProcessArgs):
    in_path: Path = arg_field(
        "--input",
        doc="The name of the folder to import data from.",
        type=FilePathType(must_exist=True),
        default=None,
    )


def tof() -> None:
    partial_args = BrukerArgs.parse_config("bruker_convert_tof")
    process_args = BrukerArgs.parse_interactive(
        "bruker_convert_tof", args=partial_args.remaining_args, exclude=["config"]
    )
    sampling = MultiLinspace(
        downsample_count=100,
        area_positions=np.array([50, 75, 100]),
        area_volumes=np.array([75, 20, 5]),
    )
    process_args.data_source = TsfDataSource(process_args.in_path, sampling=sampling)
    data_convert.process(process_args, partial_args.config)


def tims_tof() -> None:
    partial_args = BrukerArgs.parse_config("bruker_convert_tims_tof")
    process_args = BrukerArgs.parse_interactive(
        "bruker_convert_tims_tof", args=partial_args.remaining_args, exclude=["config"]
    )
    sampling = MultiLinspace(
        downsample_count=10,
        area_positions=np.array([50, 75, 100]),
        area_volumes=np.array([75, 20, 5]),
    )
    process_args.data_source = TdfDataSource(process_args.in_path, sampling=sampling)
    data_convert.process(process_args, partial_args.config)
