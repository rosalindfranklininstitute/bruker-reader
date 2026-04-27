from typing import Any
from pathlib import Path
from dataclasses import dataclass
import sys
import os

import numpy as np

from h5py import h5s

# from ms_nexus_tools.lib.utils import NotTqdm as tqdm
from tqdm import tqdm

import matplotlib.pyplot as plt

from ms_nexus_tools.api.mass_range_args import MassCentreArgs, MassRangeArgs
from ms_nexus_tools.lib.chunking import Chunker

from ms_nexus_tools.lib.nxs import (
    GenericAxis,
    Axis,
    create_field,
    NexusFile,
    FieldOptions,
)
from ms_nexus_tools.lib.bounds import ContainedBounds, Chunk, Shape, Bounds

from datargs import InteractiveArgs, ConfigFileArgs, arg_field, ArgType
from datargs.extra_types import DirPathType, FilePathType

from .loader import load_opentims


@dataclass
class ProcessArgs(InteractiveArgs, ConfigFileArgs, MassRangeArgs, MassCentreArgs):
    in_path: Path = arg_field(
        "-d",
        "--directory",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The input directory.",
        default=None,
        type=DirPathType(must_exist=True),
    )
    nxs_out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The output file.",
        default=None,
        type=FilePathType(must_exist=False),
    )

    inflate_data: bool = arg_field(
        doc="""
If present (True) the data will be inflated with zeros. The bin width will be controlled by --mass-bin-width. 
Note that this is slow becuase it expands the data by the order fo 100x.
If absent (False) the data will be stored as simply peeks. 
The data will still be a regular array with the mz dimension having size max_peaks_per_scan, 
with the leading values set to zero.
""",
        action="store_true",
    )
    mass_bin_width: float = arg_field(
        doc="The mz width of a mass bin. If not specified will use the time of flight range / 100.",
        default=None,
    )


def create_mass_axis(
    args, D, shape: Shape, chunk_shape: Shape, dtype: str, field_options: FieldOptions
):
    mass_count = shape[4]
    if args.inflate_data:
        mz_range = D.max_mz - D.min_mz
        if args.mass_bin_width is None:
            args.mass_bin_width = mz_range / mass_count
        else:
            mass_count = int(round(mz_range / args.mass_bin_width))
        mass_edges = np.array(
            [ii * args.mass_bin_width + D.min_mz for ii in range(mass_count + 1)]
        )
        mass_values = mass_edges[:-1]
        mass_axis = Axis.create(
            name="mass",
            values=mass_values,
            indices=[4],
        )
        print(
            f" Output has a mass range of {D.min_mz} - {D.max_mz}, with inflated with a bin width of {args.mass_bin_width}, giving {mass_count} mass binz."
        )
    else:
        mass_axis = Axis.create_empty(
            name="mass",
            indices=[0, 1, 2, 3, 4],
            dtype=dtype,
            shape=shape,
            compression=field_options.compression,
            compression_opts=field_options.compression_opts,
            chunks=chunk_shape,
        )
        mass_edges = None
        print(
            f" Output has a mass range of {D.min_mz} - {D.max_mz}, with stored sparse with {mass_count} mass binz."
        )

    return mass_axis, mass_edges


def read_axes(
    args, D
) -> tuple[
    GenericAxis,
    Shape,
    np.ndarray,
    dict,
    np.ndarray,
    dict,
    np.ndarray,
]:
    d = D.table2dict("MaldiFrameInfo")
    x_positions = d["XIndexPos"]
    y_positions = d["YIndexPos"]

    x_values = np.unique(x_positions)
    y_values = np.unique(y_positions)

    x_dict = {x: ii for ii, x in enumerate(x_values)}
    y_dict = {y: ii for ii, y in enumerate(y_values)}

    scan_positions = np.arange(D.min_scan, D.max_scan + 1)
    iim_edges = D.scan_to_inv_ion_mobility_frame_sorted(
        scan_positions, np.ones(scan_positions.shape)
    )
    iim_edges = np.concatenate([iim_edges, [D.min_inv_ion_mobility]])

    image_shape = (len(x_values), len(y_values), len(scan_positions))
    print(
        f" Output will be {x_values[-1] - x_values[0]} wide, over {image_shape[0]} pixels."
    )
    print(
        f" Output will be {y_values[-1] - y_values[0]} high, over {image_shape[1]} pixels."
    )
    print(f" Output has be {image_shape[2]} Inverse Ion Mobility values.")

    if args.inflate_data:
        mz_range = D.max_mz - D.min_mz
        if args.mass_bin_width is None:
            min_tof, max_tof = D.mz_to_tof([D.min_mz, D.max_mz], [1, 1])
            tof_range = max_tof - min_tof
            mass_count = int(tof_range / 100)
            args.mass_bin_width = mz_range / mass_count
        else:
            mass_count = int(round(mz_range / args.mass_bin_width))
    else:
        mass_count = int(D.GlobalMetadata["MaxNumPeaksPerScan"])

    shape = (1, *image_shape, mass_count)

    print(
        f" Output has a mass range of {D.min_mz} - {D.max_mz}, with a bin width of {args.mass_bin_width}, giving {mass_count} mass binz."
    )
    print(f" Giving a final data shape of {shape}")

    axes = GenericAxis(
        [
            [
                Axis.create(
                    name="layer",
                    values=np.array([0]),
                    indices=[0],
                )
            ],
            [
                Axis.create(
                    name="x",
                    values=x_values,
                    indices=[1],
                )
            ],
            [
                Axis.create(
                    name="y",
                    values=y_values,
                    indices=[2],
                )
            ],
            [
                Axis.create(
                    name="inv_ion_mobility",
                    values=iim_edges[:-1],
                    indices=[3],
                )
            ],
        ]
    )

    return axes, shape, x_positions, x_dict, y_positions, y_dict, iim_edges


def process(args: ProcessArgs, config: dict[str, Any]):

    with load_opentims(args.in_path) as D:
        axes, shape, x_positions, x_dict, y_positions, y_dict, iim_edges = read_axes(
            args, D
        )

        field_options = FieldOptions(
            compression="gzip",
            compression_opts=3,
            max_items_per_chunk=min(int(np.prod(shape[3:])), 1024 * 512),
            shuffle=True,
        )

        iim_edges = np.sort(iim_edges)

        max_peaks_per_scan = D.GlobalMetadata["MaxNumPeaksPerScan"]

        chunker = Chunker.from_max_item_count(
            data_shape=shape,
            priorities=(4, 3, 3, 1, 1),
            items_per_chunk=field_options.max_items_per_chunk,
        )
        mass_axis, mass_edges = create_mass_axis(
            args, D, shape, chunker.chunk_shape, "float32", field_options
        )

        axes.append([mass_axis])

        print(f" Giving a final data shape of {shape}")
        print(
            f" With a chunk shape of {chunker.chunk_shape} (cnt: {chunker.chunk_count})"
        )

        args.nxs_out_path.parent.mkdir(parents=True, exist_ok=True)

        frame_count = D.frames["NumPeaks"].size

        min_x = np.min(x_positions)
        min_y = np.min(y_positions)

        nxs = NexusFile(args.nxs_out_path, mode="w")
        with nxs.as_context() as nx_fle:
            spectra = nxs.create_subentry(
                "spectra",
                create_field(
                    dtype="uint32",
                    shape=shape,
                    compression=field_options.compression,
                    compression_opts=field_options.compression_opts,
                    chunks=chunker.chunk_shape,
                    shuffle=field_options.shuffle,
                ),
                axes=axes,
            )

            dataset = nx_fle["/entry/spectra/data/signal"]

            initial_mass = np.linspace(
                D.min_mz - 1,
                D.min_mz,
                num=shape[4],
                dtype=np.float32,
                endpoint=False,
            )

            print("Writing data:")
            for ii, frame in enumerate(
                tqdm(
                    D.query_iter(D.ms1_frames, columns=D.all_columns), total=frame_count
                )
            ):
                frame_no = frame["frame"][0]
                assert frame_no == ii + 1
                x = x_dict[x_positions[ii]]
                y = y_dict[y_positions[ii]]

                if args.inflate_data:
                    assert mass_edges is not None
                    result, _, _ = np.histogram2d(
                        frame["inv_ion_mobility"],
                        frame["mz"],
                        bins=[iim_edges, mass_edges],
                        weights=frame["intensity"],
                    )
                else:
                    data = np.vstack(
                        [
                            frame["inv_ion_mobility"],
                            frame["mz"],
                        ],
                    )

                    data.sort(axis=0)
                    iim, _ = np.histogram(data[0, :], bins=iim_edges)
                    result = np.zeros(chunker.data_shape[3:])
                    iim_end = np.cumsum(iim)
                    iim_start = np.concatenate([[0], iim_end[:-1]])

                    scan_inx = np.repeat(np.arange(len(iim)), iim)
                    data_inx = np.arange(data.shape[1])
                    start_inx = np.searchsorted(iim_start, data_inx, side="right")
                    mass_inx = data_inx - iim_start[start_inx]

                    result = np.zeros((shape[3], shape[4]), dtype=np.uint32)
                    mass_result = np.full((shape[3], shape[4]), initial_mass)

                    mass_result[scan_inx, mass_inx] = data[1, data_inx]
                    result[scan_inx, mass_inx] = frame["intensity"][data_inx]
                    nxs.root.spectra.data.mass[0, x, y, :, :] = mass_result

                nxs.root.spectra.data.signal[0, x, y, :, :] = result
                if ii > 5000:
                    break
