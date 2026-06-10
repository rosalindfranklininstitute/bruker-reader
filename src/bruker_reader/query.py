from queue import Empty, ShutDown, Queue
import itertools
import math
import time
from typing import Any, cast, Generic, TypeVar
from bisect import bisect_left, bisect_right
from pathlib import Path
from dataclasses import dataclass, asdict
import tracemalloc, linecache, os
from tracemalloc import Statistic, StatisticDiff, Snapshot
import concurrent.futures as cfutures

from datargs.config_args import ConfigFileArgs
from datargs.interactive_args import InteractiveArgs
from datargs.extra_types import FilePathType, DirPathType
from datargs.args import arg_field

from ms_nexus_tools.lib.utils import slice_range, format_bytes
from ms_nexus_tools.lib.nxs import NexusFile, NxAxes, NxAxis, create_field
from ms_nexus_tools.lib.chunker import Chunker
from ms_nexus_tools.lib.bounds import Chunk, Shape
from ms_nexus_tools.lib.plotting import Plottable
from ms_nexus_tools.lib.image import XYRectangle, plot_image as ms_plot_image
from ms_nexus_tools.lib.spectrum import SpecSlice, plot_spectrum

import hdf5plugin

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

from icecream import ic

from .maldi import MZSpectraType, Metadata

Float1D32 = np.ndarray[tuple[int], np.dtype[np.float32]]
Int1D32 = np.ndarray[tuple[int], np.dtype[np.int32]]


snapshot: Snapshot | None = None


def push_snapshot():
    global snapshot
    if False:
        new_snapshot = tracemalloc.take_snapshot()
        new_snapshot = new_snapshot.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            )
        )
        display_top(new_snapshot.statistics("lineno"))
        if snapshot is not None:
            display_diff(new_snapshot.compare_to(snapshot, "lineno"))
        snapshot = new_snapshot


def display_top(top_stats: list[Statistic], limit=5):

    print(" ===== ")
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %s"
            % (
                index,
                Path(*Path(frame.filename).parts[-3:]),
                frame.lineno,
                format_bytes(stat.size),
            )
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %s" % (len(other), format_bytes(size)))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %s" % format_bytes(total))


def display_diff(top_stats: list[StatisticDiff], limit=5):

    print(" >> Diff %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %s"
            % (
                index,
                Path(*Path(frame.filename).parts[-3:]),
                frame.lineno,
                format_bytes(stat.size_diff),
            )
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size_diff = sum(stat.size_diff for stat in other)
        print("%s other: %s" % (len(other), format_bytes(size_diff)))
    total_diff = sum(stat.size_diff for stat in top_stats)
    print("Total allocated size: %s" % format_bytes(total_diff))


def _slice_from_axis(start: float, stop: float, values: Float1D32) -> slice:
    start_index = bisect_left(values, start)
    stop_index = bisect_right(values, stop)
    return slice(start_index, stop_index)


@dataclass
class Titled:
    title: str


@dataclass
class MZSlice:
    mz_start: float
    mz_stop: float

    def mz_slice(self, mz_values: Float1D32) -> slice:
        return _slice_from_axis(self.mz_start, self.mz_stop, mz_values)

    def mz_mask(self, mz_values: np.ndarray) -> tuple[np.ndarray, ...]:
        return np.nonzero((mz_values >= self.mz_start) & (mz_values < self.mz_stop))


class MZSliceAlaise(SpecSlice):
    def __init__(self, mz_slice: MZSlice):
        self.start = mz_slice.mz_start
        self.stop = mz_slice.mz_stop


@dataclass
class IIMSlice:
    iim_start: float
    iim_stop: float

    def iim_slice(self, iim_values: Float1D32) -> slice:
        return _slice_from_axis(self.iim_start, self.iim_stop, iim_values)


class IIMSliceAlaise(SpecSlice):
    def __init__(self, iim_slice: IIMSlice):
        self.start = iim_slice.iim_start
        self.stop = iim_slice.iim_stop


class MzIIMSelect(Titled, MZSlice, IIMSlice):
    def __init__(
        self,
        title: str,
        mz_start: float,
        mz_stop: float,
        iim_start: float,
        iim_stop: float,
    ):
        self.title = title
        self.mz_start = mz_start
        self.mz_stop = mz_stop
        self.iim_start = iim_start
        self.iim_stop = iim_stop

    @staticmethod
    def read_csv(filename: Path) -> list["MzIIMSelect"]:
        return []

    def to_title(self) -> str:
        return f"1/k0: [{self.iim_start}-{self.iim_stop}), mz: [{self.mz_start},{self.mz_stop})"


class MzXYSelect(Titled, MZSlice, XYRectangle):
    def __init__(
        self,
        title: str,
        mz_start: float,
        mz_stop: float,
        x_start: float,
        x_stop: float,
        y_start: float,
        y_stop: float,
    ):
        self.title = title
        self.mz_start = mz_start
        self.mz_stop = mz_stop
        self.x_start = x_start
        self.x_stop = x_stop
        self.y_start = y_start
        self.y_stop = y_stop

    @staticmethod
    def read_csv(filename: Path) -> list["MzXYSelect"]:
        return []

    def to_title(self) -> str:
        return f"x: [{self.x_start}-{self.x_stop}) y: [{self.y_start}-{self.y_stop}), mz: [{self.mz_start},{self.mz_stop})"


class MzIIMXYSelect(Titled, MZSlice, IIMSlice, XYRectangle):
    def __init__(
        self,
        title: str,
        mz_start: float,
        mz_stop: float,
        iim_start: float,
        iim_stop: float,
        x_start: float,
        x_stop: float,
        y_start: float,
        y_stop: float,
    ):
        self.title = title
        self.mz_start = mz_start
        self.mz_stop = mz_stop
        self.iim_start = iim_start
        self.iim_stop = iim_stop
        self.x_start = x_start
        self.x_stop = x_stop
        self.y_start = y_start
        self.y_stop = y_stop

    @staticmethod
    def read_csv(filename: Path) -> list["MzIIMXYSelect"]:
        return []

    def to_title(self) -> str:
        return f"x: [{self.x_start}-{self.x_stop}) y: [{self.y_start}-{self.y_stop}), 1/k0: [{self.iim_start}-{self.iim_stop}), mz: [{self.mz_start},{self.mz_stop})"


@dataclass
class ProcessArgs(ConfigFileArgs, InteractiveArgs):
    in_path: Path = arg_field(
        "--input",
        type=FilePathType(must_exist=True),
        doc="The input nxs file to read data from",
        required=True,
        default=None,
    )

    out_path: Path = arg_field(
        "--output",
        type=DirPathType(must_exist=False),
        doc="The output folder to write to.",
        required=True,
        default=None,
    )

    dry_run: bool = arg_field(
        action="store_true",
        doc="If true the full dataset will not be read. Only the total images and the selectors will be plotted. Useful for checking that the ranges are correct.",
    )

    mz_iim_selection: list[list[str]] = arg_field(
        "--mz-iim",
        doc="The Mz and 1/K0 values to sum into each pixel. This requires five values.",
        nargs=5,
        action="append",
        default_factory=list,
        metavar=("Name", "mz_start", "mz_end", "1/ko_start", "1/ko_end"),
    )

    mz_iim_csv: Path = arg_field(
        doc="A csv file to read mz iim selctions from. Each line is one selection.",
        type=FilePathType(must_exist=True),
        default=None,
    )

    mz_xy_selection: list[list[str]] = arg_field(
        "--mz-xy",
        doc="The Mz, X and Y values to sum into each 1/ko vs mz spectrum. This requires seven values.",
        nargs=7,
        action="append",
        default_factory=list,
        metavar=("Name", "mz_start", "mz_end", "x_start", "x_end", "y_start", "y_end"),
    )

    mz_xy_csv: Path = arg_field(
        doc="A csv file to read mz xy selctions from. Each line is one selection.",
        type=FilePathType(must_exist=True),
        default=None,
    )

    mz_iim_xy_selection: list[list[str]] = arg_field(
        "--mz-xy-iim",
        doc="The Mz, 1/K0, X and Y values to sum. This requires 9 values.",
        nargs=9,
        action="append",
        default_factory=list,
        metavar=(
            "Name",
            "mz_start",
            "mz_end",
            "iim_start",
            "iim_end",
            "x_start",
            "x_end",
            "y_start",
            "y_end",
        ),
    )

    mz_iim_xy_csv: Path = arg_field(
        doc="A csv file to read mz iim xy selctions from. Each line is one selection.",
        type=FilePathType(must_exist=True),
        default=None,
    )


def plot_image(
    image,
    x_values,
    y_values,
    xy_slices: list[Plottable[XYRectangle]],
    title: str,
    sub_title: str,
    out_path: Path,
    diff_selector=np.median,
):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_title(sub_title)

    im, (im_min, im_max) = ms_plot_image(
        ax,
        image=image,
        x_values=x_values,
        y_values=y_values,
        xy_rectangles=xy_slices,
        diff_selector=diff_selector,
        cmap="viridis",
        origin="lower",
    )
    fig.colorbar(
        im,
        ax=ax,
        location="right",
        shrink=0.8,
        ticks=np.linspace(im_min, im_max, 6),
    )
    fig.savefig(out_path / f"xy-{title}.png")
    plt.close(fig)


def plot_iim(
    iim_values,
    counts,
    iim_slices: list[Plottable[IIMSlice]],
    title: str,
    sub_title: str,
    out_path: Path,
):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_title(sub_title)

    plot_spectrum(
        ax,
        counts,
        iim_values,
        [Plottable(p.title, p.color, IIMSliceAlaise(p.value)) for p in iim_slices],
    )

    ax.set_xlabel("1/K0")
    ax.set_ylabel("Ion Count")
    fig.savefig(out_path / f"iim-{title}.png")
    plt.close(fig)


def plot_mz(
    mz_values,
    counts,
    mz_slices: list[Plottable[MZSlice]],
    title: str,
    sub_title: str,
    out_path: Path,
):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_title(sub_title)

    plot_spectrum(
        ax,
        counts,
        mz_values,
        [Plottable(p.title, p.color, MZSliceAlaise(p.value)) for p in mz_slices],
    )

    ax.set_xlabel("mz")
    ax.set_ylabel("Ion Count")
    fig.savefig(out_path / f"mz-{title}.png")
    plt.close(fig)


def calculate_memory_chunks(
    data_shape: Shape, data_chunk_shape: Shape, max_memory: int
) -> Chunker:
    chunk_memory = np.prod(data_chunk_shape) * 4
    chunks_per_memory = Chunker.from_max_item_count(
        tuple([ds // cs for ds, cs in zip(data_shape[1:3], data_chunk_shape[1:3])]),
        priorities=(1, 2),
        items_per_chunk=int(math.floor(max_memory / (chunk_memory * 2))),
    )

    memory = Chunker.from_chunk_shape(
        data_shape,
        Shape(
            [
                data_chunk_shape[0],
                data_chunk_shape[1] * chunks_per_memory.chunk_shape[0],
                data_chunk_shape[2] * chunks_per_memory.chunk_shape[1],
                *data_chunk_shape[3:],
            ]
        ),
    )
    return memory


def read_chunk(
    in_path: Path,
    chunk: Chunk,
    data_queue: Queue,
    start: float = -1,
    cutoff: float = -1,
):
    if start > 0 and cutoff > 0:
        now = time.monotonic()
        if (now - start) > cutoff:
            return
    nxs = NexusFile(
        in_path,
        "r",
    )
    chunk_mz = nxs.root.spectra.data.mass[*chunk].nxdata
    chunk_count = nxs.root.spectra.data.signal[*chunk].nxdata
    nxs._file.close()
    data_queue.put((chunk, chunk_mz, chunk_count))


@dataclass
class Selections:
    mz_xy_indices: dict[tuple[int, int], list[int]]
    mz_xy: list[MzXYSelect]

    mz_iim: list[MzIIMSelect]
    mz_iim_slices: list[slice]

    mz_iim_xy_indices: dict[tuple[int, int], list[int]]
    mz_iim_xy_slices: list[slice]
    mz_iim_xy: list[MzIIMXYSelect]


@dataclass
class Results:
    xy_count: int
    images: np.ndarray[tuple[int, int, int]]
    iims: np.ndarray[tuple[int, int]]
    totals: np.ndarray[tuple[int]]


def process_chunk(
    selections: Selections,
    results: Results,
    data_queue: Queue,
    chunk_total: int,
    xy_total: int,
):
    outer_tqdm = tqdm(desc="Processing", total=chunk_total)
    inner_tqdm = tqdm(desc="XY", total=xy_total)
    while True:
        try:
            file_chunk, chunk_mz, chunk_count = data_queue.get(timeout=30)
        except Empty:
            print("Empty")
            continue
        except ShutDown:
            print("Shutdown")
            return

        total_xy = [
            xy
            for xy in itertools.product(
                slice_range(file_chunk[1]), slice_range(file_chunk[2])
            )
        ]

        for xy in total_xy:
            results.xy_count += 1
            x, y = xy
            x_ii = x - file_chunk[1].start
            y_ii = y - file_chunk[2].start
            sel_mz = chunk_mz[0, x_ii, y_ii, :, :]
            sel_count = chunk_count[0, x_ii, y_ii, :, :]

            if xy in selections.mz_xy_indices:
                for ii in selections.mz_xy_indices[xy]:
                    mz_xy_sel = selections.mz_xy[ii]
                    mz_mask = mz_xy_sel.mz_mask(sel_mz)
                    count_masked = sel_count[mz_mask]
                    try:
                        results.iims[mz_mask[0], ii] += count_masked
                    except Exception as e:
                        ic("mz_xy", ii)
                        ic((mz_xy_sel.mz_start, mz_xy_sel.mz_stop))
                        ic(sel_mz.shape)
                        ic(sel_count.shape)
                        ic(mz_mask)
                        ic(count_masked)
                        ic(results.iims.shape)
                        ic(e)
                        raise

            if xy in selections.mz_iim_xy_indices:
                for ii in selections.mz_iim_xy_indices[xy]:
                    mz_iim_xy_sel = selections.mz_iim_xy[ii]
                    sub_sel_mz = sel_mz[selections.mz_iim_xy_slices[ii], :]
                    sub_sel_count = sel_count[selections.mz_iim_xy_slices[ii], :]
                    mz_mask = mz_iim_xy_sel.mz_mask(sub_sel_mz)
                    results.totals[ii] += np.sum(sub_sel_count[mz_mask])

            for ii, mz_iim_sel in enumerate(selections.mz_iim):
                sub_sel_mz = sel_mz[selections.mz_iim_slices[ii], :]
                sub_sel_count = sel_count[selections.mz_iim_slices[ii], :]
                mz_mask = mz_iim_sel.mz_mask(sub_sel_mz)
                results.images[x, y, ii] = np.sum(sub_sel_count[mz_mask])
            inner_tqdm.update()
        outer_tqdm.update()
        data_queue.task_done()


def plot_totals(
    out_path: Path,
    x_values,
    y_values,
    tic_image,
    iim_values,
    iim_spectrum,
    mz_values,
    mz_spectrum,
    selections: Selections,
):
    color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS)

    mz_xy_plottables = [
        Plottable(sel.title, next(color_cycle), sel) for sel in selections.mz_xy
    ]
    mz_iim_plottables = [
        Plottable(sel.title, next(color_cycle), sel) for sel in selections.mz_iim
    ]
    mz_iim_xy_plottables = [
        Plottable(sel.title, next(color_cycle), sel) for sel in selections.mz_iim_xy
    ]

    plot_image(
        tic_image,
        x_values,
        y_values,
        cast(list[Plottable[XYRectangle]], [*mz_xy_plottables, *mz_iim_xy_plottables]),
        "Total Image",
        "",
        out_path,
    )
    plot_iim(
        iim_values,
        iim_spectrum,
        cast(list[Plottable[IIMSlice]], [*mz_iim_plottables, *mz_iim_xy_plottables]),
        "Total Iim",
        "",
        out_path,
    )
    plot_mz(
        mz_values,
        mz_spectrum,
        cast(
            list[Plottable[MZSlice]],
            [*mz_iim_plottables, *mz_xy_plottables, *mz_iim_xy_plottables],
        ),
        "Total Mz",
        "",
        out_path,
    )


def process(args: ProcessArgs, config: dict[str, Any] = {}):
    tracemalloc.start()

    assert args.in_path.exists(), f"The input file {args.in_path} was not found"
    args.out_path.mkdir(parents=True, exist_ok=True)
    selections = Selections({}, [], [], [], {}, [], [])

    selections.mz_iim = [
        MzIIMSelect(v[0], *[float(vi) for vi in v[1:]]) for v in args.mz_iim_selection
    ]
    selections.mz_iim.extend(MzIIMSelect.read_csv(args.mz_iim_csv))

    selections.mz_xy = [
        MzXYSelect(v[0], *[float(vi) for vi in v[1:]]) for v in args.mz_xy_selection
    ]
    selections.mz_xy.extend(MzXYSelect.read_csv(args.mz_xy_csv))

    selections.mz_iim_xy = [
        MzIIMXYSelect(v[0], *[float(vi) for vi in v[1:]])
        for v in args.mz_iim_xy_selection
    ]
    selections.mz_iim_xy.extend(MzIIMXYSelect.read_csv(args.mz_iim_xy_csv))

    push_snapshot()

    nxs = NexusFile(
        args.in_path,
        "r",
    )
    with nxs.as_context() as nx_fle:
        metadata = Metadata(MZSpectraType.PEAKS, -1, -1, -1, -1, -1, -1, -1, -1)
        md_dict = asdict(metadata)
        for key, value in nxs.instrument.attrs.items():
            if key in md_dict:
                md_dict[key] = value
        metadata = Metadata(**md_dict)

        x_values = nxs.root.spectra.data.x.nxdata
        y_values = nxs.root.spectra.data.y.nxdata
        iim_values = nxs.root.spectra.data.inv_ion_mobility.nxdata
        iim_spectrum = nxs.root.mobiligram.data.signal[0, :].nxdata
        mz_values = nxs.root.mz_spectrum.data.mass.nxdata
        mz_spectrum = nxs.root.mz_spectrum.data.signal[0, :].nxdata
        tic_image = nxs.root.tic.data.signal[0, :, :].nxdata

        nx = len(x_values)
        ny = len(y_values)

        for sel in selections.mz_iim:
            selections.mz_iim_slices.append(sel.iim_slice(iim_values))

        for ii, sel in enumerate(selections.mz_xy):
            for xy in itertools.product(
                slice_range(sel.x_slice(x_values)), slice_range(sel.y_slice(y_values))
            ):
                if xy not in selections.mz_xy_indices:
                    selections.mz_xy_indices[xy] = []
                selections.mz_xy_indices[xy].append(ii)

        selections.mz_iim_xy_slices = [
            sel.iim_slice(iim_values) for sel in selections.mz_iim_xy
        ]

        for ii, sel in enumerate(selections.mz_iim_xy):
            for xy in itertools.product(
                slice_range(sel.x_slice(x_values)), slice_range(sel.y_slice(y_values))
            ):
                if xy not in selections.mz_iim_xy_indices:
                    selections.mz_iim_xy_indices[xy] = []
                selections.mz_iim_xy_indices[xy].append(ii)

        results = Results(
            xy_count=0,
            images=np.zeros((nx, ny, len(selections.mz_iim))),
            iims=np.zeros((len(iim_values), len(selections.mz_xy))),
            totals=np.zeros((len(selections.mz_iim_xy))),
        )

        data_shape = nxs.root.spectra.data.signal.shape
        data_chunk_shape = nxs.root.spectra.data.signal.chunks
        max_memory = 1024 * 1024 * 1024
        n_producers = 2

        memory = calculate_memory_chunks(data_shape, data_chunk_shape, max_memory)

        print(f" Using maximum memory of ({format_bytes(max_memory)}).")
        print(
            f" Data has a shape {data_shape} and a chunk shape of {data_chunk_shape},"
        )
        print(
            f" and a chunk size of {np.prod(data_chunk_shape)} items ({format_bytes(np.prod(data_chunk_shape) * 4)})."
        )
        chunk_memory = np.prod(memory.chunk_shape) * 4
        print(
            f" Using a memory chunk shape of {memory.chunk_shape} with {np.prod(memory.chunk_shape)} items ({format_bytes(chunk_memory)}*2 = {format_bytes(chunk_memory * 2)}),"
        )
    push_snapshot()

    plot_totals(
        args.out_path,
        x_values,
        y_values,
        tic_image,
        iim_values,
        iim_spectrum,
        mz_values,
        mz_spectrum,
        selections,
    )

    if args.dry_run:
        return

    data_queue = Queue(maxsize=n_producers)

    start = time.monotonic()

    chunks = [c for c in memory.chunks()]
    xy_total_count = np.prod(data_shape[:3])
    with cfutures.ThreadPoolExecutor(max_workers=n_producers + 1) as executor:
        processor_future = executor.submit(
            process_chunk, selections, results, data_queue, len(chunks), xy_total_count
        )

        reader_futures = [
            executor.submit(read_chunk, args.in_path, c, data_queue, start, -1)
            for c in chunks
        ]

        for f in tqdm(
            cfutures.as_completed(reader_futures),
            total=len(reader_futures),
            desc="  Chunks ",
            leave=False,
        ):
            f.result()

        data_queue.join()
        data_queue.shutdown()

        cfutures.wait([processor_future])
        processor_future.result()

    stop = time.monotonic()
    print(
        f"Processed {results.xy_count} of {xy_total_count} ({results.xy_count * 100 / xy_total_count:.1f}%) pixels"
    )
    print(f" in {stop - start:.2f} seconds.")
    print(f" Giving {results.xy_count / (stop - start):.1f} pixels per second.")

    out_nxs = NexusFile(args.out_path / "outputs.nxs", "w")
    out_nxs.create_subentry(
        "TotalImage",
        create_field(value=tic_image),
        NxAxes(
            [
                [NxAxis.create(name="x", values=x_values, indices=[0])],
                [NxAxis.create(name="y", values=y_values, indices=[1])],
            ]
        ),
    )

    for ii, mz_iim_sel in enumerate(selections.mz_iim):
        plot_image(
            results.images[:, :, ii],
            x_values,
            y_values,
            [],
            mz_iim_sel.title,
            mz_iim_sel.to_title(),
            args.out_path,
        )
        out_nxs.create_subentry(
            f"Image - {mz_iim_sel.title}",
            create_field(value=results.images[:, :, ii]),
            NxAxes(
                [
                    [NxAxis.create(name="x", values=x_values, indices=[0])],
                    [NxAxis.create(name="y", values=y_values, indices=[1])],
                ]
            ),
        )

    out_nxs.create_subentry(
        "InverseIonMobility",
        create_field(value=iim_spectrum),
        NxAxes(
            [
                [NxAxis.create(name="iim", values=iim_values, indices=[0])],
            ]
        ),
    )

    for ii, mz_xy_sel in enumerate(selections.mz_xy):
        plot_iim(
            iim_values,
            results.iims[:, ii],
            [],
            mz_xy_sel.title,
            mz_xy_sel.to_title(),
            args.out_path,
        )
        out_nxs.create_subentry(
            f"IIM - {mz_xy_sel.title}",
            create_field(value=results.iims[:, ii]),
            NxAxes(
                [
                    [NxAxis.create(name="iim", values=iim_values, indices=[0])],
                ]
            ),
        )

    out_nxs.create_subentry(
        "Mz",
        create_field(value=mz_spectrum),
        NxAxes(
            [
                [NxAxis.create(name="mz", values=mz_values, indices=[0])],
            ]
        ),
    )

    out_nxs.create_subentry(
        "Totals",
        create_field(value=[results.totals]),
        NxAxes(
            [
                [
                    NxAxis.create(
                        name="slice",
                        values=[[sel.title for sel in selections.mz_iim_xy]],
                        indices=[1],
                    )
                ],
            ]
        ),
    )

    for ii, mz_iim_xy_sel in enumerate(selections.mz_iim_xy):
        print(f"{ii}: {mz_iim_xy_sel.title}: {results.totals[ii]}")
