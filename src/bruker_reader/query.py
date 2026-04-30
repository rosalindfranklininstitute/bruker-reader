import itertools
from typing import Any, Union, Protocol, cast, Generic, TypeVar
from bisect import bisect_left, bisect_right
from pathlib import Path
from dataclasses import dataclass, asdict

from datargs.config_args import ConfigFileArgs
from datargs.interactive_args import InteractiveArgs
from datargs.extra_types import FilePathType, DirPathType
from datargs.args import arg_field

from ms_nexus_tools.lib.utils import slice_range
from ms_nexus_tools.lib.nxs import NexusFile

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

from icecream import ic

from .maldi import MZSpectraType, Metadata

Float1D32 = np.ndarray[tuple[int], np.dtype[np.float32]]
Int1D32 = np.ndarray[tuple[int], np.dtype[np.int32]]


def _slice_from_axis(start: float, stop: float, values: Float1D32) -> slice:
    start_index = bisect_left(values, start)
    stop_index = bisect_right(values, stop)
    return slice(start_index, stop_index)


@dataclass
class Titled:
    title: str


@dataclass
class XYRectangle:
    x_start: float
    x_stop: float
    y_start: float
    y_stop: float

    def x_slice(self, x_values: Float1D32) -> slice:
        return _slice_from_axis(self.x_start, self.x_stop, x_values)

    def y_slice(self, y_values: Float1D32) -> slice:
        return _slice_from_axis(self.y_start, self.y_stop, y_values)

    def get_plot_rect(self, **kwargs) -> Rectangle:
        x = self.x_start
        w = self.x_stop - x
        y = self.y_start
        h = self.y_stop - y
        return Rectangle((x, y), w, h, **kwargs)


@dataclass
class MZSlice:
    mz_start: float
    mz_stop: float

    def mz_slice(self, mz_values: Float1D32) -> slice:
        return _slice_from_axis(self.mz_start, self.mz_stop, mz_values)

    def mz_mask(self, mz_values: np.ndarray) -> np.ndarray:
        return np.nonzero((mz_values >= self.mz_start) & (mz_values < self.mz_stop))


@dataclass
class IIMSlice:
    iim_start: float
    iim_stop: float

    def iim_slice(self, iim_values: Float1D32) -> slice:
        return _slice_from_axis(self.iim_start, self.iim_stop, iim_values)


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

    mz_bin_width: float = arg_field(
        default=None,
        doc="If specified this will be used as the width of the mz bins used to plot the mz spectrum over the whole image.",
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


T = TypeVar("T")


@dataclass
class Plottable(Generic[T]):
    title: str
    color: str
    value: T


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
    im_min, im_max = np.percentile(image, [0, 100])
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    mnx = np.min(x_values)
    mxx = np.max(x_values)
    mny = np.min(y_values)
    mxy = np.max(y_values)
    dx = diff_selector(np.diff(x_values))
    dy = diff_selector(np.diff(y_values))
    img, xedges, yedges = np.histogram2d(
        xx.ravel(),
        yy.ravel(),
        weights=image.ravel(),
        bins=[
            np.arange(mnx - dx / 2, mxx + dx / 2, dx),
            (np.arange(mny - dy / 2, mxy + dy / 2, dy)),
        ],
    )
    im = ax.imshow(
        img.T,
        cmap="inferno",
        extent=(mnx, mxx, mny, mxy),
        origin="lower",
    )
    for jj, mz_xy_rect in enumerate(xy_slices):
        rect = mz_xy_rect.value.get_plot_rect(
            linewidth=2,
            edgecolor=mz_xy_rect.color,
            facecolor=mz_xy_rect.color,
            alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            *rect.get_bbox().max,
            mz_xy_rect.title,
            color=mz_xy_rect.color,
            fontsize=12,
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
    ax.plot(iim_values, counts)

    for ii, iim_slice in enumerate(iim_slices):
        x = iim_slice.value.iim_start
        w = iim_slice.value.iim_stop - iim_slice.value.iim_start
        rect = Rectangle(
            (x, 0),
            width=w,
            height=1,
            transform=ax.get_xaxis_transform(),
            linewidth=2,
            edgecolor=iim_slice.color,
            facecolor=iim_slice.color,
            alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            1.01,
            iim_slice.title,
            transform=ax.get_xaxis_transform(),
            fontsize=12,
            color=iim_slice.color,
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
    ax.plot(mz_values, counts)

    for ii, mz_slice in enumerate(mz_slices):
        x = mz_slice.value.mz_start
        w = mz_slice.value.mz_stop - mz_slice.value.mz_start
        rect = Rectangle(
            (x, 0),
            width=w,
            height=1,
            transform=ax.get_xaxis_transform(),
            linewidth=2,
            edgecolor=mz_slice.color,
            facecolor=mz_slice.color,
            alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            1.01,
            mz_slice.title,
            transform=ax.get_xaxis_transform(),
            fontsize=12,
            color=mz_slice.color,
        )

    ax.set_xlabel("mz")
    ax.set_ylabel("Ion Count")
    fig.savefig(out_path / f"mz-{title}.png")
    plt.close(fig)


def process(args: ProcessArgs, config: dict[str, Any] = {}):

    assert args.in_path.exists(), f"The input file {args.in_path} was not found"
    args.out_path.mkdir(parents=True, exist_ok=True)

    mz_iim_selections = [
        MzIIMSelect(v[0], *[float(vi) for vi in v[1:]]) for v in args.mz_iim_selection
    ]
    mz_iim_selections.extend(MzIIMSelect.read_csv(args.mz_iim_csv))

    mz_xy_selections = [
        MzXYSelect(v[0], *[float(vi) for vi in v[1:]]) for v in args.mz_xy_selection
    ]
    mz_xy_selections.extend(MzXYSelect.read_csv(args.mz_xy_csv))

    mz_iim_xy_selections = [
        MzIIMXYSelect(v[0], *[float(vi) for vi in v[1:]])
        for v in args.mz_iim_xy_selection
    ]
    mz_iim_xy_selections.extend(MzIIMXYSelect.read_csv(args.mz_iim_xy_csv))

    nxs = NexusFile(args.in_path, "r")
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

        nx = len(x_values)
        ny = len(y_values)

        if args.mz_bin_width is not None:
            min_mz = 50
            max_mz = 1200
            bins = int(np.ceil((max_mz - min_mz) / args.mz_bin_width))
            mz_edges = np.linspace(min_mz, max_mz, endpoint=True, num=bins)
            mz_values = mz_edges[1:]
            mz_spectrum = np.zeros(mz_values.shape)

        mz_iim_slices: list[slice] = []
        for sel in mz_iim_selections:
            mz_iim_slices.append(sel.iim_slice(iim_values))

        mz_xy_indices: dict[tuple[int, int], list[int]] = {}
        for ii, sel in enumerate(mz_xy_selections):
            for xy in itertools.product(
                slice_range(sel.x_slice(x_values)), slice_range(sel.y_slice(y_values))
            ):
                if xy not in mz_xy_indices:
                    mz_xy_indices[xy] = []
                mz_xy_indices[xy].append(ii)

        mz_iim_xy_slices: list[slice] = [
            sel.iim_slice(iim_values) for sel in mz_iim_xy_selections
        ]
        mz_iim_xy_starts: Float1D32 = np.array(
            [sel.iim_start for sel in mz_iim_xy_selections]
        )

        mz_iim_xy_indices: dict[tuple[int, int], list[int]] = {}
        for ii, sel in enumerate(mz_iim_xy_selections):
            for xy in itertools.product(
                slice_range(sel.x_slice(x_values)), slice_range(sel.y_slice(y_values))
            ):
                if xy not in mz_iim_xy_indices:
                    mz_iim_xy_indices[xy] = []
                mz_iim_xy_indices[xy].append(ii)

        total_xy = [xy for xy in itertools.product(range(nx), range(ny))]

        images = np.zeros((nx, ny, len(mz_iim_selections)))
        iims = np.zeros((len(iim_values), len(mz_iim_selections)))
        totals = np.zeros((len(mz_iim_xy_selections)))

        for xy_ii, xy in enumerate(tqdm(total_xy, smoothing=0.1)):
            x, y = xy
            sel_mz = nxs.root.spectra.data.mass[0, x, y, :, :]
            sel_count = nxs.root.spectra.data.signal[0, x, y, :, :]

            if args.mz_bin_width is not None:
                spec, _ = np.histogram(
                    sel_mz.ravel(), mz_edges, weights=sel_count.ravel()
                )
                mz_spectrum += spec

            if xy in mz_xy_indices:
                for pix, (mz, count) in enumerate(zip(sel_mz, sel_count)):
                    for ii in mz_xy_indices[xy]:
                        mz_xy_sel = mz_xy_selections[ii]
                        mz_slice = mz_xy_sel.mz_slice(mz)
                        iims[pix, ii] += np.sum(count[mz_slice])

            if xy in mz_iim_xy_indices:
                for ii in mz_iim_xy_indices[xy]:
                    mz_iim_xy_sel = mz_iim_xy_selections[ii]
                    sub_sel_mz = sel_mz[mz_iim_xy_slices[ii], :]
                    sub_sel_count = sel_count[mz_iim_xy_slices[ii], :]
                    mz_mask = mz_iim_xy_sel.mz_mask(sub_sel_mz)
                    totals[ii] += np.sum(sub_sel_count[mz_mask])

            for ii, mz_iim_sel in enumerate(mz_iim_selections):
                sub_sel_mz = sel_mz[mz_iim_slices[ii], :]
                sub_sel_count = sel_count[mz_iim_slices[ii], :]
                mz_mask = mz_iim_sel.mz_mask(sub_sel_mz)
                images[x, y, ii] = np.sum(sub_sel_count[mz_mask])

        color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS)

        mz_xy_plottables = [
            Plottable(sel.title, next(color_cycle), sel) for sel in mz_xy_selections
        ]
        mz_iim_plottables = [
            Plottable(sel.title, next(color_cycle), sel) for sel in mz_iim_selections
        ]
        mz_iim_xy_plottables = [
            Plottable(sel.title, next(color_cycle), sel) for sel in mz_iim_xy_selections
        ]

        plot_image(
            nxs.root.tic.data.signal[0, :, :].nxdata,
            x_values,
            y_values,
            cast(
                list[Plottable[XYRectangle]], [*mz_xy_plottables, *mz_iim_xy_plottables]
            ),
            "Total Image",
            "",
            args.out_path,
        )
        for ii, mz_iim_sel in enumerate(mz_iim_selections):
            plot_image(
                images[:, :, ii],
                x_values,
                y_values,
                [],
                mz_iim_sel.title,
                mz_iim_sel.to_title(),
                args.out_path,
            )

        plot_iim(
            iim_values,
            nxs.root.mobiligram.data.signal[0, :].nxdata,
            cast(
                list[Plottable[IIMSlice]], [*mz_iim_plottables, *mz_iim_xy_plottables]
            ),
            "Total Iim",
            "",
            args.out_path,
        )

        for ii, mz_xy_sel in enumerate(mz_xy_selections):
            plot_iim(
                iim_values,
                iims[:, ii],
                [],
                mz_xy_sel.title,
                mz_xy_sel.to_title(),
                args.out_path,
            )

        if args.mz_bin_width is not None:
            plot_mz(
                mz_values,
                mz_spectrum,
                cast(
                    list[Plottable[MZSlice]],
                    [*mz_iim_plottables, *mz_xy_plottables, *mz_iim_xy_plottables],
                ),
                "Total Mz",
                "",
                args.out_path,
            )

        for ii, mz_iim_xy_sel in enumerate(mz_iim_xy_selections):
            print(f"{ii}: {mz_iim_xy_sel.title}: {totals[ii]}")
