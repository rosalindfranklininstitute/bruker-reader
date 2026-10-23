import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from icecream import ic
from pyTDFSDK.classes import TdfData
from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
from pyTDFSDK.tims import (
    tims_index_to_mz,
    tims_read_scans_v2,
    tims_scannum_to_oneoverk0,
)
from xartus.lib.bounds import Chunk, Shape
from xartus.lib.data_source import (
    AbstractDataSource,
    Axis,
    AxisType,
    DataShape,
    MultiCOO,
    Signal,
)
from xartus.lib.dtypes import Float1D32, Int1D32
from xartus.lib.multi_linspace import MultiLinspace

from bruker_reader.utils import MaldiAxis


class TdfDataSource(AbstractDataSource):
    def __init__(self, tdf_file: Path, sampling: MultiLinspace | None = None):
        if tdf_file.suffix != ".tdf":
            raise ValueError(
                f"Expected the path to an analysis.tdf file, but received '{tdf_file}'"
            )
        self.dll = init_tdf_sdk_api()
        self.tof_data = TdfData(
            bruker_d_folder_name=str(tdf_file.parent), tdf_sdk=self.dll
        )

        min_mz = self.tof_data.GlobalMetadata["MzAcqRangeLower"]
        max_mz = self.tof_data.GlobalMetadata["MzAcqRangeUpper"]
        min_iim = self.tof_data.GlobalMetadata["OneOverK0AcqRangeLower"]
        max_iim = self.tof_data.GlobalMetadata["OneOverK0AcqRangeUpper"]

        mass_count = self.tof_data.GlobalMetadata["DigitizerNumSamples"]
        peak_range, scan_range, id_range = self.tof_data.analysis.range(
            "Frames", ["NumPeaks", "NumScans", "Id"]
        )
        self.min_peaks, self.max_peaks = peak_range
        self.min_scans, self.max_scans = scan_range
        self.frame_count = np.diff(id_range)[0]

        self.max_data_count = self.max_peaks * self.frame_count

        ic(min_mz, max_mz)
        ic(min_iim, max_iim)
        ic(mass_count)

        sampling = MultiLinspace() if sampling is None else sampling
        self.mz_edges = sampling.get_edges(min_mz, max_mz, mass_count)
        ic(len(self.mz_edges))
        scan_positions = np.arange(0, self.max_scans)
        self.iim_edges = tims_scannum_to_oneoverk0(
            tdf_sdk=self.dll,
            handle=self.tof_data.handle,
            frame_id=id_range[0],
            scan_nums=scan_positions,
        )

        self.frame_info = self.tof_data.analysis.join_frame("MaldiFrameInfo")
        self.maldi_axis = MaldiAxis(self.frame_info)

        self.total_shape = (
            len(self.maldi_axis.x_values),
            len(self.maldi_axis.y_values),
            len(self.iim_edges),
            len(self.mz_edges) - 1,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tof_data.close()

    def instrument_metadata(self) -> dict[str, Any]:
        names = [
            "AcquisitionDateTime",
            "AcquisitionFirmwareVersion",
            "AcquisitionSoftware",
            "AcquisitionSoftwareVendor",
            "AcquisitionSoftwareVersion",
            "AnalysisId",
            "ClosedProperly",
            "DenoisingEnabled",
            "Description",
            "DigitizerFullScale",
            "DigitizerNumSamples",
            "DigitizerSerialNumber",
            "DigitizerType",
            "Geometry",
            "InstrumentFamily",
            "InstrumentName",
            "InstrumentRevision",
            "InstrumentSerialNumber",
            "InstrumentSourceType",
            "InstrumentVendor",
            "MaldiApplicationType",
            "MethodName",
            "OperatorName",
            "PeakListIndexScaleFactor",
            "PeakWidthEstimateType",
            "PeakWidthEstimateValue",
            "RunId",
            "SampleName",
            "SchemaType",
            "SchemaVersionMajor",
            "SchemaVersionMinor",
            "TargetId",
            "TimsCompressionType",
        ]

        return {name: self.tof_data.GlobalMetadata[name] for name in names}

    def experiment_metadata(self) -> dict[str, Any]:
        return {}

    def shape(self) -> DataShape:
        total_data_capacity = np.prod(self.total_shape)
        density = self.max_data_count / total_data_capacity
        ic(self.total_shape, self.max_data_count, total_data_capacity)
        ic(density)
        if density > 1.0:
            raise ValueError(
                f"The predicted density ({density:.2f} is greater than 1.0. This is likely because the number of mass bins is too small."
            )

        return DataShape(
            shape=self.total_shape, is_sparse=True, worst_case_density=density
        )

    def signal_definition(self) -> Signal:
        return Signal("signal", np.int64, "counts")

    def output_chunks(self) -> dict[str, Shape]:
        return {"spectra": (1, 1, 1, 1)}

    def read_chunks(self) -> list[Shape] | None:
        return [(3, 3, 2, 1)]

    def chunk_read_count(self, memory_chunk: Shape) -> int:
        return memory_chunk[0] * memory_chunk[1]

    def axis_definitions(self) -> list[Axis]:
        return [
            Axis(
                name="x",
                primary_axis=0,
                axis_type=AxisType.EXACT,
                units="m",
                dtype=np.float32,
            ),
            Axis(
                name="y",
                primary_axis=1,
                axis_type=AxisType.EXACT,
                units="m",
                dtype=np.float32,
            ),
            Axis(
                name="inv_ion_mobility",
                primary_axis=2,
                axis_type=AxisType.EXACT,
                units="1/k0",
                dtype=np.float32,
            ),
            Axis(
                name="mz",
                primary_axis=3,
                axis_type=AxisType.BINNED,
                units="mz",
                dtype=np.float32,
            ),
        ]

    def exact_axis_values(self, axis: Axis) -> np.ndarray:
        match axis.name:
            case "x":
                return self.maldi_axis.x_values
            case "y":
                return self.maldi_axis.y_values
            case "inv_ion_mobility":
                return self.iim_edges
            case _:
                raise ValueError(f"Unknown continuous axis requested: {axis.name}")

    def binned_axis_edges(self, axis: Axis) -> np.ndarray:
        if axis.name != "mz":
            raise ValueError(f"Unknown binned axis requested: {axis.name}")
        return self.mz_edges

    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        return {
            "total_image": ("inv_ion_mobility", "mz"),
            "mz_spectra": ("x", "y", "inv_ion_mobility"),
            "mobiligram": ("x", "y", "mz"),
        }

    def fill_chunk(
        self,
        memory_chunk: Chunk,
        update: Callable[[int], None],
    ) -> np.ndarray | MultiCOO:

        if memory_chunk.shape[-1] != self.total_shape[-1]:
            raise NotImplementedError(
                "The final dimension is always read in full. But a short memory chunk was provided."
            )

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
                frame_data = tims_read_scans_v2(
                    tdf_sdk=self.dll,
                    handle=self.tof_data.handle,
                    frame_id=frame_inx,
                    scan_begin=memory_chunk[2].start,
                    scan_end=memory_chunk[2].stop,
                )
                for ii_scan, (index_array, intensity_array) in enumerate(frame_data):
                    count = len(index_array)
                    counts.append(count)
                    coords.append([ii_x, ii_y, ii_scan, 0])
                    int_data.append(intensity_array)
                    inx_data.append(index_array)

            update(1)

        if len(inx_data) == 0:
            return MultiCOO(
                coords=np.empty(shape=(4, 0), dtype=np.int32),
                values={"signal": np.array([]), "mz": np.array([])},
            )

        inx_array = np.concatenate(inx_data)
        mz_array = tims_index_to_mz(
            tdf_sdk=self.dll,
            handle=self.tof_data.handle,
            frame_id=last_frame_inx,
            indices=inx_array,
        )
        labels = np.searchsorted(self.mz_edges[1:], mz_array)
        labels[labels == self.total_shape[-1]] = self.total_shape[-1] - 1
        final_coords = np.array(coords).T
        final_coords = np.repeat(final_coords, counts, axis=1)
        final_coords[3, :] = labels

        # unique = np.apply_along_axis(lambda row: np.unique(row).size, 1, final_coords)
        # total = np.array([s for s in memory_chunk.shape])
        # dimension_density = unique / total
        # overall_density = ic(final_coords.shape[1]) / ic(np.prod(memory_chunk.shape))
        # ic(dimension_density, overall_density)

        return MultiCOO(
            coords=final_coords,
            values={"signal": np.concatenate(int_data), "mz": mz_array},
        )
