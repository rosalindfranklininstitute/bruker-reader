from enum import StrEnum
from matplotlib.image import NonUniformImage
from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict
import sys
import os

from threading import Lock, local
import concurrent.futures as cfutures

import numpy as np

import hdf5plugin

# from ms_nexus_tools.lib.utils import NotTqdm as tqdm
from tqdm import tqdm

import matplotlib.pyplot as plt

from ms_nexus_tools.lib.utils import format_bytes
from ms_nexus_tools.api.mass_range_args import MassCentreArgs, MassRangeArgs
from ms_nexus_tools.lib.chunker import Chunker

from ms_nexus_tools.lib.nxs import (
    NxAxes,
    NxAxis,
    create_field,
    NexusFile,
    FieldOptions,
)
from ms_nexus_tools.lib.bounds import Chunk, Shape, Bounds

from datargs import InteractiveArgs, ConfigFileArgs, arg_field, ArgType
from datargs.extra_types import DirPathType, FilePathType

from .loader import LoadOpenTims


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
        mass_axis = NxAxis.create(
            name="mass",
            values=mass_values,
            indices=[4],
        )
        print(
            f" Output has a mass range of {D.min_mz} - {D.max_mz}, with inflated with a bin width of {args.mass_bin_width}, giving {mass_count} mass binz."
        )
    else:
        mass_axis = NxAxis.create_empty(
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
    NxAxes,
    Shape,
    np.ndarray,
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

    frame_x_to_axis = {fx: ii for ii, fx in enumerate(x_values)}
    frame_y_to_axis = {fy: ii for ii, fy in enumerate(y_values)}

    frame_inx = np.full((len(x_values), len(y_values)), -1)
    for ii, (x, y) in enumerate(zip(frame_x, frame_y)):
        ii_x = frame_x_to_axis[x]
        ii_y = frame_y_to_axis[y]
        assert frame_inx[ii_x, ii_y] == -1
        frame_inx[ii_x, ii_y] = ii + 1

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

    axes = NxAxes(
        [
            [
                NxAxis.create(
                    name="layer",
                    values=np.array([0]),
                    indices=[0],
                )
            ],
            [
                NxAxis.create(
                    name="x",
                    values=x_values,
                    indices=[1],
                )
            ],
            [
                NxAxis.create(
                    name="y",
                    values=y_values,
                    indices=[2],
                )
            ],
            [
                NxAxis.create(
                    name="inv_ion_mobility",
                    values=iim_edges[1:],
                    indices=[3],
                )
            ],
        ]
    )

    return (
        axes,
        shape,
        frame_inx,
        frame_x,
        frame_x_to_axis,
        frame_y,
        frame_y_to_axis,
        iim_edges,
    )


Int1D32 = np.ndarray[tuple[int], np.dtype[np.int32]]
Int2D32 = np.ndarray[tuple[int, int], np.dtype[np.int32]]
Float1D32 = np.ndarray[tuple[int], np.dtype[np.float32]]


@dataclass
class IndexingData:
    frame_inx: Int2D32
    frame_x: Int1D32
    frame_y: Int1D32
    frame_x_to_axis: dict[int, int]
    frame_y_to_axis: dict[int, int]
    shape: Shape
    chunker: Chunker
    initial_mass: Float1D32
    iim_edges: Float1D32
    mz_edges: Float1D32


def fill_memory_block(
    args,
    D,
    meta: IndexingData,
    chunk: Chunk,
    tic_values,
    mobiligram_values,
    mz_spectrum,
) -> tuple[np.ndarray, np.ndarray]:
    mz_buffer = np.zeros(chunk.shape, dtype=np.float32)
    int_buffer = np.zeros(chunk.shape, dtype=np.uint32)

    frames = [ii for ii in meta.frame_inx[chunk[1], chunk[2]].ravel() if ii >= 0]

    for frame in tqdm(
        D.query_iter(frames, columns=D.all_columns),
        total=len(frames),
        desc="frames in chunk",
        leave=False,
    ):
        ii = frame["frame"][0] - 1
        assert np.all(frame["frame"] == ii + 1)

        x = meta.frame_x_to_axis[meta.frame_x[ii]]
        y = meta.frame_y_to_axis[meta.frame_y[ii]]

        buffer_x = x - chunk[1].start
        buffer_y = y - chunk[2].start
        assert buffer_x < chunk[1].stop
        assert buffer_y < chunk[2].stop
        assert buffer_x >= 0
        assert buffer_y >= 0

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

        iim_bin_inx = np.digitize(data[1, :], bins=meta.iim_edges) - 1
        assert np.all(iim_bin_inx >= 0)
        assert np.all(iim_bin_inx < len(meta.iim_edges) - 1)

        tmp = np.bincount(
            iim_bin_inx, weights=intensity, minlength=len(meta.iim_edges) - 1
        )
        tmp = tmp.astype(np.uint32)
        mobiligram_values += tmp
        mz_v, _ = np.histogram(data[0, :], bins=meta.mz_edges, weights=intensity)
        mz_spectrum += mz_v

        if args.inflate_mz:
            assert meta.mz_edges is not None
            result, _, _ = np.histogram2d(
                data[1, :],
                data[0, :],
                bins=[meta.iim_edges, meta.mz_edges],
                weights=intensity,
            )
        else:
            iim = np.bincount(iim_bin_inx, minlength=len(meta.iim_edges) - 1)
            result = np.zeros(meta.chunker.data_shape[3:])
            iim_end = np.cumsum(iim)
            iim_start = np.concatenate([[0], iim_end[:-1]])

            scan_inx = np.repeat(np.arange(len(iim)), iim)
            data_inx = np.arange(len(intensity))
            start_inx = np.searchsorted(iim_start, data_inx, side="right")
            mass_inx = data_inx - iim_start[start_inx]

            result = np.zeros((meta.shape[3], meta.shape[4]), dtype=np.uint32)
            mass_result = np.full((meta.shape[3], meta.shape[4]), meta.initial_mass)

            mass_result[scan_inx, mass_inx] = data[0, :]
            result[scan_inx, mass_inx] = intensity

            mz_buffer[0, buffer_x, buffer_y, :, :] = mass_result
            # nxs.root.spectra.data.mass[0, x, y, :, :] = mass_result

        int_buffer[0, buffer_x, buffer_y, :, :] = result
        # nxs.root.spectra.data.signal[0, x, y, :, :] = result
    return mz_buffer, int_buffer


def write_to_nxs(
    nxs: NexusFile, memory_chunk: Chunk, mz_buffer: np.ndarray, int_buffer: np.ndarray
):
    nxs.root.spectra.data.mass[*memory_chunk] = mz_buffer[...]
    nxs.root.spectra.data.signal[*memory_chunk] = int_buffer[...]


def process(args: ProcessArgs, config: dict[str, Any]):

    with LoadOpenTims(args.in_path) as D:
        (
            axes,
            shape,
            frame_inx,
            frame_x,
            frame_x_to_axis,
            frame_y,
            frame_y_to_axis,
            iim_edges,
        ) = read_axes(args, D)

        field_options = FieldOptions(
            compression=hdf5plugin.Blosc(),
            compression_opts=None,
            max_items_per_chunk=1024 * 1024 * 8,
            shuffle=True,
        )
        chunker = Chunker.from_max_item_count(
            data_shape=shape,
            priorities=(4, 3, 3, 2, 1),
            items_per_chunk=field_options.max_items_per_chunk,
        )

        memory_buffer_multiplier = 3

        memory = Chunker.from_chunk_shape(
            data_shape=shape,
            chunk_shape=Shape(
                [
                    chunker.chunk_shape[0],
                    chunker.chunk_shape[1],
                    chunker.chunk_shape[2],
                    *[c * memory_buffer_multiplier for c in chunker.chunk_shape[3:]],
                ]
            ),
        )

        initial_mass = np.linspace(
            D.min_mz - 1,
            D.min_mz,
            num=shape[4],
            dtype=np.float32,
            endpoint=False,
        )
        mass_bin_width = get_mass_bin_width(args, D)
        mass_count = int(round((D.max_mz - D.min_mz) / mass_bin_width))
        mz_edges = np.linspace(D.min_mz, D.max_mz, endpoint=True, num=mass_count + 1)
        meta = IndexingData(
            frame_inx=frame_inx,
            frame_x=frame_x,
            frame_y=frame_y,
            frame_x_to_axis=frame_x_to_axis,
            frame_y_to_axis=frame_y_to_axis,
            shape=shape,
            chunker=chunker,
            initial_mass=initial_mass,
            iim_edges=iim_edges,
            mz_edges=mz_edges,
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
        print(
            f" Using a memory chunk shape of {memory.chunk_shape} with {np.prod(memory.chunk_shape)} items ({format_bytes(np.prod(memory.chunk_shape) * 4 * 2)}),"
        )

        args.nxs_out_path.parent.mkdir(parents=True, exist_ok=True)

        nxs = NexusFile(args.nxs_out_path, mode="w")
        with nxs.as_context():
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
                axes=NxAxes([axes[0], axes[1], axes[2]]),
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
                axes=NxAxes([axes[0], [axes[3][0].copy_with_incremented_indices(-2)]]),
            )

            nxs.create_subentry(
                "mz_spectrum",
                create_field(
                    dtype="uint32",
                    shape=(1, len(meta.mz_edges[1:])),
                    compression=field_options.compression,
                    compression_opts=field_options.compression_opts,
                    chunks=(1, len(meta.mz_edges[1:])),
                    shuffle=field_options.shuffle,
                ),
                axes=NxAxes(
                    [
                        axes[0],
                        [
                            NxAxis.create(
                                name="mass",
                                values=meta.mz_edges[1:],
                                indices=[1],
                            )
                        ],
                    ]
                ),
            )

            mobiligram_values = np.zeros((shape[3]), dtype=np.uint32)
            tic_values = np.zeros(shape[1:3], dtype=np.uint32)

            mz_spectrum = np.zeros(meta.mz_edges[1:].shape, dtype=np.float64)

            nexus_file_lock = Lock()
            experiment_file_lock = Lock()

            def read_chunk(memory_chunk):
                local_store = local()
                with experiment_file_lock:
                    local_store.mz_buffer, local_store.int_buffer = fill_memory_block(
                        args,
                        D,
                        meta,
                        memory_chunk,
                        tic_values,
                        mobiligram_values,
                        mz_spectrum,
                    )
                with nexus_file_lock:
                    write_to_nxs(
                        nxs, memory_chunk, local_store.mz_buffer, local_store.int_buffer
                    )

            outer_chunks = [chunk for chunk in memory.chunks()]
            # for memory_chunk in tqdm(
            #     outer_chunks,
            #     total=len(outer_chunks),
            #     desc="  Chunks ",
            #     leave=False,
            # ):
            #     read_chunk(memory_chunk)
            print("Writing data:")
            with cfutures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(read_chunk, memory_chunk)
                    for memory_chunk in outer_chunks
                ]

                for memory_chunk in tqdm(
                    cfutures.as_completed(futures),
                    total=len(outer_chunks),
                    desc="  Chunks ",
                    leave=False,
                ):
                    pass

            nxs.root.tic.data.signal[0, :, :] = tic_values
            nxs.root.mobiligram.data.signal[0, :] = mobiligram_values
            nxs.root.mz_spectrum.data.signal[0, :] = mz_spectrum
