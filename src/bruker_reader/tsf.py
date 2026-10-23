# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pyTDFSDK.classes import TsfData
from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
from pyTDFSDK.tsf import (
    tsf_index_to_mz,
    tsf_read_line_spectrum_v2,
)
from xartus.lib import (
    AbstractDataSource,
    Axis,
    AxisType,
    DataShape,
    MultiCOO,
    MultiLinspace,
    Signal,
)
from xartus.lib.bounds import Chunk, Shape
from xartus.lib.dtypes import Float1D32, Int1D32

from bruker_reader.utils import MaldiAxis

logger = logging.getLogger(__name__)


class TsfDataSource(AbstractDataSource):
    def __init__(self, tsf_file: Path, sampling: MultiLinspace | None):
        if tsf_file.suffix != ".tsf":
            raise ValueError(
                f"Expected the path to an analysis.tsf file, but received '{tsf_file}'"
            )
        self.dll = init_tdf_sdk_api()
        self.tof_data = TsfData(
            bruker_d_folder_name=str(tsf_file.parent), tdf_sdk=self.dll
        )

        min_mz = self.tof_data.GlobalMetadata["MzAcqRangeLower"]
        max_mz = self.tof_data.GlobalMetadata["MzAcqRangeUpper"]
        mass_count = self.tof_data.GlobalMetadata["DigitizerNumSamples"]
        ranges = self.tof_data.analysis.range("Frames", ["NumPeaks", "Id"])
        self.min_peaks, self.max_peaks = ranges[0]
        self.frame_count = np.diff(ranges[1])[0]

        self.max_data_count = self.max_peaks * self.frame_count

        sampling = MultiLinspace() if sampling is None else sampling
        self.mz_edges = sampling.get_edges(min_mz, max_mz, mass_count)

        self.frame_info = self.tof_data.analysis.join_frame("MaldiFrameInfo")
        self.maldi_axis = MaldiAxis(self.frame_info)

        self.total_shape = (
            len(self.maldi_axis.x_values),
            len(self.maldi_axis.y_values),
            len(self.mz_edges) - 1,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tof_data.close()

    def instrument_metadata(self) -> dict[str, Any]:
        names = [
            "SchemaType",
            "SchemaVersionMajor",
            "SchemaVersionMinor",
            "AcquisitionSoftwareVendor",
            "InstrumentVendor",
            "ClosedProperly",
            "TimsCompressionType",
            "AnalysisId",
            "DigitizerNumSamples",
            "AcquisitionSoftware",
            "AcquisitionSoftwareVersion",
            "AcquisitionFirmwareVersion",
            "AcquisitionDateTime",
            "InstrumentName",
            "InstrumentFamily",
            "InstrumentRevision",
            "InstrumentSourceType",
            "InstrumentSerialNumber",
            "OperatorName",
            "Description",
            "SampleName",
            "MethodName",
            "DenoisingEnabled",
            "PeakWidthEstimateValue",
            "PeakWidthEstimateType",
            "HasLineSpectra",
            "HasLineSpectraPeakWidth",
            "HasProfileSpectra",
            "MaldiApplicationType",
            "RunId",
            "TargetId",
            "Geometry",
            "DigitizerType",
            "DigitizerSerialNumber",
            "DigitizerFullScale",
        ]

        return {name: self.tof_data.GlobalMetadata[name] for name in names}

    def experiment_metadata(self) -> dict[str, Any]:
        return {}

    def shape(self) -> DataShape:
        total_data_capacity = np.prod(self.total_shape)
        density = self.max_data_count / total_data_capacity
        if density > 1.0:
            logger.warning(
                f"The predicted density ({density:.2f} is greater than 1.0. This is likely because the number of mass bins is too small."
            )

        return DataShape(
            shape=self.total_shape, is_sparse=True, worst_case_density=density
        )

    def signal_definition(self) -> Signal:
        return Signal("signal", np.int64, "counts")

    def output_chunks(self) -> dict[str, Shape]:
        return {"images": (1, 1, 2), "spectra": (2, 2, 1)}

    def chunk_read_count(self, memory_chunk: Shape) -> int:
        return memory_chunk[0] * memory_chunk[1]

    def axis_definitions(self) -> list[Axis]:
        return [
            Axis(
                name="x",
                primary_axis=0,
                axis_type=AxisType.EXACT,
                units="m",
                dtype=np.int64,
            ),
            Axis(
                name="y",
                primary_axis=1,
                axis_type=AxisType.EXACT,
                units="m",
                dtype=np.int64,
            ),
            Axis(
                name="mz",
                primary_axis=2,
                axis_type=AxisType.BINNED,
                units="mz",
                dtype=np.float64,
            ),
        ]

    def exact_axis_values(self, axis: Axis) -> np.ndarray:
        match axis.name:
            case "x":
                return self.maldi_axis.x_values
            case "y":
                return self.maldi_axis.y_values
            case _:
                raise ValueError(f"Unknown continuous axis requested: {axis.name}")

    def binned_axis_edges(self, axis: Axis) -> np.ndarray:
        if axis.name != "mz":
            raise ValueError(f"Unknown binned axis requested: {axis.name}")
        return self.mz_edges

    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        return {"total_image": ("mz",), "total_spectra": ("x", "y")}

    def fill_chunk(
        self,
        memory_chunk: Chunk,
        update: Callable[[int], None],
    ) -> np.ndarray | MultiCOO:

        coords: list[list[int]] = []
        counts: list[int] = []
        int_data: list[Int1D32] = []
        inx_data: list[Float1D32] = []
        last_frame_inx = 0

        for ii_x, ii_y in itertools.product(
            memory_chunk.range(0), memory_chunk.range(1)
        ):
            frame_inx = self.maldi_axis.frame_inx[ii_x, ii_y]
            if frame_inx >= 0:
                last_frame_inx = frame_inx
                index_array, intensity_array = tsf_read_line_spectrum_v2(
                    tdf_sdk=self.dll, handle=self.tof_data.handle, frame_id=frame_inx
                )
                count = len(index_array)
                counts.append(count)
                coords.append([ii_x, ii_y, 0])
                int_data.append(intensity_array)
                inx_data.append(index_array)

            update(1)

        inx_array = np.concatenate(inx_data)
        mz_array = tsf_index_to_mz(
            tdf_sdk=self.dll,
            handle=self.tof_data.handle,
            frame_id=last_frame_inx,
            indices=inx_array,
        )
        labels = np.searchsorted(self.mz_edges[1:], mz_array)
        labels[labels == self.total_shape[-1]] = self.total_shape[-1] - 1
        final_coords = np.array(coords).T
        final_coords = np.repeat(final_coords, counts, axis=1)
        final_coords[2, :] = labels

        return MultiCOO(
            coords=final_coords,
            values={"signal": np.concatenate(int_data), "mz": mz_array},
        )
