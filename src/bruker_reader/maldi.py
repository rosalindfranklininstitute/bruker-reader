from enum import StrEnum
from matplotlib.image import NonUniformImage
from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict
import sys
import os

import numpy as np

import hdf5plugin

# from ms_nexus_tools.lib.utils import NotTqdm as tqdm
from tqdm import tqdm

import matplotlib.pyplot as plt

from ms_nexus_tools.lib.utils import format_bytes
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


class MZSpectraType(StrEnum):
    PEAKS = "peaks"
    CONTINUOUS = "continuous"


@dataclass
class Metadata:
    mz_spectra_type: MZSpectraType
    min_mz: float
    max_mz: float
    min_inv_ion_mobility: float
    max_inv_ion_mobility: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass
class ProcessArgs(InteractiveArgs, ConfigFileArgs):
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

    inflate_mz: bool = arg_field(
        "--inflate",
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


def read_metadata(args, D) -> Metadata:
    return Metadata(
        mz_spectra_type=MZSpectraType.CONTINUOUS
        if args.inflate_mz
        else MZSpectraType.PEAKS,
        min_mz=D.min_mz,
        max_mz=D.max_mz,
        min_inv_ion_mobility=D.min_inv_ion_mobility,
        max_inv_ion_mobility=D.max_inv_ion_mobility,
        min_x=D.GlobalMetadata["ImagingAreaMinXIndexPos"],
        max_x=D.GlobalMetadata["ImagingAreaMaxXIndexPos"],
        min_y=D.GlobalMetadata["ImagingAreaMinYIndexPos"],
        max_y=D.GlobalMetadata["ImagingAreaMaxYIndexPos"],
    )


def get_mass_bin_width(args, D):
    if args.mass_bin_width is None:
        mz_range = D.max_mz - D.min_mz
        min_tof, max_tof = D.mz_to_tof([D.min_mz, D.max_mz], [1, 1])
        tof_range = max_tof - min_tof
        mass_count = int(tof_range / 100)
        return mz_range / mass_count
    else:
        return args.mass_bin_width


def get_mass_count(args, D) -> int:
    if args.inflate_mz:
        mz_range = D.max_mz - D.min_mz
        if args.mass_bin_width is None:
            min_tof, max_tof = D.mz_to_tof([D.min_mz, D.max_mz], [1, 1])
            tof_range = max_tof - min_tof
            return int(tof_range / 100)
        else:
            return int(round(mz_range / args.mass_bin_width))
    else:
        return int(D.GlobalMetadata["MaxNumPeaksPerScan"])


def create_mass_axis(
    args, D, shape: Shape, chunk_shape: Shape, dtype: str, field_options: FieldOptions
):
    mass_count = shape[4]
    if args.inflate_mz:
        mz_range = D.max_mz - D.min_mz
        if args.mass_bin_width is None:
            args.mass_bin_width = mz_range / mass_count
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
    frame_x = d["XIndexPos"]
    frame_y = d["YIndexPos"]

    x_values = np.unique(frame_x)
    y_values = np.unique(frame_y)

    x_dict = {x: ii for ii, x in enumerate(x_values)}
    y_dict = {y: ii for ii, y in enumerate(y_values)}

    scan_positions = np.arange(D.min_scan, D.max_scan + 1)
    iim_edges = D.scan_to_inv_ion_mobility_frame_sorted(
        scan_positions, np.ones(scan_positions.shape)
    )
    iim_edges = np.sort(np.concatenate([iim_edges, [D.min_inv_ion_mobility]]))

    image_shape = (len(x_values), len(y_values), len(scan_positions))
    print(
        f" Output will be {x_values[-1] - x_values[0]} wide, over {image_shape[0]} pixels."
    )
    print(
        f" Output will be {y_values[-1] - y_values[0]} high, over {image_shape[1]} pixels."
    )
    print(f" Output has be {image_shape[2]} Inverse Ion Mobility values.")

    mass_count = get_mass_count(args, D)

    shape = (1, *image_shape, mass_count)

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
                    values=iim_edges[1:],
                    indices=[3],
                )
            ],
        ]
    )

    return axes, shape, frame_x, x_dict, frame_y, y_dict, iim_edges


def process(args: ProcessArgs, config: dict[str, Any]):

    with load_opentims(args.in_path) as D:
        axes, shape, frame_x, x_dict, frame_y, y_dict, iim_edges = read_axes(args, D)

        d = D.table2dict("MaldiFrameInfo")
        frame_x = d["XIndexPos"]
        frame_y = d["YIndexPos"]

        field_options = FieldOptions(
            compression=hdf5plugin.Blosc(),
            compression_opts=None,
            max_items_per_chunk=min(int(np.prod(shape[3:])), 1024 * 512),
            shuffle=True,
        )

        chunker = Chunker.from_max_item_count(
            data_shape=shape,
            priorities=(4, 3, 3, 2, 1),
            items_per_chunk=field_options.max_items_per_chunk,
        )
        memory = Chunker.from_max_item_count(
            data_shape=shape,
            priorities=(4, 3, 3, 2, 1),
            items_per_chunk=1024 * 1024 * 256,
        )
        mass_axis, mass_edges = create_mass_axis(
            args, D, shape, chunker.chunk_shape, "float32", field_options
        )

        axes.append([mass_axis])

        print(f" Giving a final data shape of {shape}")
        print(
            f" With a chunk shape of {chunker.chunk_shape} and a chunk count of {chunker.chunk_count},"
        )
        print(
            f" and a chunk size of {np.prod(chunker.chunk_shape)} items ({format_bytes(np.prod(chunker.chunk_shape) * 4)})."
        )

        args.nxs_out_path.parent.mkdir(parents=True, exist_ok=True)

        frame_count = D.frames["NumPeaks"].size

        nxs = NexusFile(args.nxs_out_path, mode="w")
        with nxs.as_context() as nx_fle:
            metadata = read_metadata(args, D)
            for key, value in asdict(metadata).items():
                nxs.instrument.attrs[key] = value

            nxs.create_subentry(
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
            nxs.create_subentry(
                "tic",
                create_field(
                    dtype="uint32",
                    shape=shape[:3],
                    compression=field_options.compression,
                    compression_opts=field_options.compression_opts,
                    chunks=(1, *shape[1:3]),
                    shuffle=field_options.shuffle,
                ),
                axes=GenericAxis([axes[1], axes[2]]),
            )

            nxs.create_subentry(
                "mobiligram",
                create_field(
                    dtype="uint32",
                    shape=(1, shape[3]),
                    compression=field_options.compression,
                    compression_opts=field_options.compression_opts,
                    chunks=(1, shape[3]),
                    shuffle=field_options.shuffle,
                ),
                axes=GenericAxis([axes[3]]),
            )

            mass_bin_width = get_mass_bin_width(args, D)
            mass_count = int(round((D.max_mz - D.min_mz) / mass_bin_width))
            nxs.create_subentry(
                "mz_spectrum",
                create_field(
                    dtype="uint32",
                    shape=(1, mass_count),
                    compression=field_options.compression,
                    compression_opts=field_options.compression_opts,
                    chunks=(1, mass_count),
                    shuffle=field_options.shuffle,
                ),
                axes=GenericAxis([axes[3]]),
            )

            initial_mass = np.linspace(
                D.min_mz - 1,
                D.min_mz,
                num=shape[4],
                dtype=np.float32,
                endpoint=False,
            )

            mobiligram_values = np.zeros((shape[3]), dtype=np.uint32)
            tic_values = np.zeros(shape[1:3], dtype=np.uint32)

            mz_edges = np.linspace(
                D.min_mz, D.max_mz, endpoint=True, num=mass_count + 1
            )
            mz_spectrum = np.zeros(mz_edges[1:].shape, dtype=np.float64)

            print("Writing data:")
            for ii, frame in enumerate(
                tqdm(
                    D.query_iter(D.ms1_frames, columns=D.all_columns), total=frame_count
                )
            ):
                assert np.all(frame["frame"] == ii + 1)
                x = x_dict[frame_x[ii]]
                y = y_dict[frame_y[ii]]

                tic_values[x, y] = D.frames["SummedIntensities"][ii]

                data = np.vstack(
                    [
                        frame["mz"],
                        frame["inv_ion_mobility"],
                    ],
                )
                sort_inx = np.lexsort(data)  # lexsort sorts last column first.
                intensity = frame["intensity"][sort_inx]
                data = data[:, sort_inx]

                iim_bin_inx = np.digitize(data[1, :], bins=iim_edges) - 1
                assert np.all(iim_bin_inx >= 0)
                assert np.all(iim_bin_inx < len(iim_edges) - 1)

                mobiligram_values = np.bincount(
                    iim_bin_inx, weights=intensity, minlength=len(iim_edges) - 1
                )

                mz_v, _ = np.histogram(data[0, :], bins=mz_edges, weights=intensity)
                mz_spectrum += mz_v

                if args.inflate_mz:
                    assert mass_edges is not None
                    result, _, _ = np.histogram2d(
                        data[1, :],
                        data[0, :],
                        bins=[iim_edges, mass_edges],
                        weights=intensity,
                    )
                else:
                    iim = np.bincount(iim_bin_inx, minlength=len(iim_edges) - 1)
                    result = np.zeros(chunker.data_shape[3:])
                    iim_end = np.cumsum(iim)
                    iim_start = np.concatenate([[0], iim_end[:-1]])

                    scan_inx = np.repeat(np.arange(len(iim)), iim)
                    data_inx = np.arange(len(intensity))
                    start_inx = np.searchsorted(iim_start, data_inx, side="right")
                    mass_inx = data_inx - iim_start[start_inx]

                    result = np.zeros((shape[3], shape[4]), dtype=np.uint32)
                    mass_result = np.full((shape[3], shape[4]), initial_mass)

                    mass_result[scan_inx, mass_inx] = data[0, :]
                    result[scan_inx, mass_inx] = intensity
                    nxs.root.spectra.data.mass[0, x, y, :, :] = mass_result

                nxs.root.spectra.data.signal[0, x, y, :, :] = result
                if ii > 500:
                    break
            nxs.root.tic.data.signal[0, :, :] = tic_values
            nxs.root.mobiligram.data.signal[0, :] = mobiligram_values
            nxs.root.mz_spectrum.data.signal[0, :] = mz_spectrum
