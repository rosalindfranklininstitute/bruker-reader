import itertools
from typing import Callable, Any
from pathlib import Path

import numpy as np
import numpy.typing as npt
import sparse
from icecream import ic


from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
from pyTDFSDK.classes import TsfData, TdfData
from pyTDFSDK.tsf import (
    tsf_read_line_spectrum_v2,
    tsf_index_to_mz,
    tsf_mz_to_index,
)

from ms_nexus_tools.lib.bounds import Shape, Chunk
from ms_nexus_tools.lib.data_source import (
    AbstractDataSource,
    Axis,
    AxisDensity,
    DataShape,
    MultiCOO,
)
from ms_nexus_tools.lib.dtypes import Int3D32, Int1D32, Float1D32

from bruker_reader.utils import SparseAxisSampling, MaldiAxis


class TsfDataSource(AbstractDataSource):
    def __init__(
        self, tsf_file: Path, sampling: SparseAxisSampling = SparseAxisSampling()
    ):
        if tsf_file.suffix != ".tsf":
            raise ValueError(
                f"Expected the path to an analysis.tsf file, but recived '{tsf_file}'"
            )
        self.dll = init_tdf_sdk_api()
        self.tof_data = TsfData(
            bruker_d_folder_name=str(tsf_file.parent), tdf_sdk=self.dll
        )

        min_mz = self.tof_data.GlobalMetadata["MzAcqRangeLower"]
        max_mz = self.tof_data.GlobalMetadata["MzAcqRangeUpper"]
        mass_count = (
            self.tof_data.GlobalMetadata["DigitizerNumSamples"]
            // sampling.downsample_count
        )
        ranges = self.tof_data.analysis.range("Frames", ["NumPeaks", "Id"])
        self.min_peaks, self.max_peaks = ranges[0]
        self.frame_count = np.diff(ranges[1])[0]

        self.max_data_count = self.max_peaks * self.frame_count

        ends = np.concatenate(
            [[min_mz], (max_mz - min_mz) * sampling.area_positions / 100.0 + min_mz]
        )
        self.mz_edges = np.concatenate(
            [
                *[
                    np.linspace(
                        ends[ii],
                        ends[ii + 1],
                        num=int(mass_count * sampling.area_volumes[ii] / 100.0),
                        endpoint=False,
                    )
                    for ii in range(len(sampling.area_positions))
                ],
                [max_mz],
            ]
        )

        self.frame_info = self.tof_data.analysis.join_frame("MaldiFrameInfo")
        self.maldi_axis = MaldiAxis(self.frame_info)

        self.total_shape = (
            len(self.maldi_axis.x_values),
            len(self.maldi_axis.y_values),
            len(self.mz_edges) - 1,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self.tof_data.close()

    def instrament_metadata(self) -> dict[str, Any]:
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
            raise ValueError(
                f"The predicted density ({density:.2f} is greater than 1.0. This is likely because the number of mass bins is too small."
            )

        return DataShape(shape=self.total_shape, density=density)

    def signal_type(self) -> npt.DTypeLike:
        return np.int64

    def output_chunks(self) -> dict[str, Shape]:
        return dict(images=(1, 1, 2), spectra=(2, 2, 1))

    def chunk_read_count(self, memory_chunk: Shape) -> int:
        return memory_chunk[0] * memory_chunk[1]

    def axis_definitions(self) -> list[Axis]:
        return [
            Axis(
                name="x",
                primary_axis=0,
                secondary_axes=[],
                density=AxisDensity.CONTINUOUS,
                units="m",
                dtype=np.float32,
            ),
            Axis(
                name="y",
                primary_axis=1,
                secondary_axes=[],
                density=AxisDensity.CONTINUOUS,
                units="m",
                dtype=np.float32,
            ),
            Axis(
                name="mz",
                primary_axis=2,
                secondary_axes=[0, 1],
                density=AxisDensity.SPARSE,
                units="mz",
                dtype=np.float32,
            ),
        ]

    def continuous_axis_values(self, axis: Axis) -> np.ndarray:
        match axis.name:
            case "x":
                return self.maldi_axis.x_values
            case "y":
                return self.maldi_axis.y_values
            case _:
                raise ValueError(f"Unknown continuous axis requested: {axis.name}")

    def sparse_axis_edges(self, axis: Axis) -> np.ndarray:
        if axis.name != "mz":
            raise ValueError(f"Unknown sparse axis requested: {axis.name}")
        return self.mz_edges

    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        return dict(total_image=("mz",), total_spectra=("x", "y"))

    def fill_chunk(
        self,
        memory_chunk: Chunk,
        fill_axis: list[Axis],
        update: Callable[[int], None],
    ) -> np.ndarray | sparse.COO:

        assert len(fill_axis) == 1
        assert fill_axis[0].name == "mz"

        coords: list[Int3D32] = []
        data: list[Int1D32] = []
        mz_data: list[Float1D32] = []

        for ii_x, ii_y in itertools.product(
            memory_chunk.range(0), memory_chunk.range(1)
        ):
            frame_inx = self.maldi_axis.frame_inx[ii_x, ii_y]
            if frame_inx >= 0:
                index_array, intensity_array = tsf_read_line_spectrum_v2(
                    tdf_sdk=self.dll, handle=self.tof_data.handle, frame_id=frame_inx
                )
                mz_array = tsf_index_to_mz(
                    tdf_sdk=self.dll,
                    handle=self.tof_data.handle,
                    frame_id=frame_inx,
                    indices=index_array,
                )

                count = len(index_array)

                frame_coord = np.array([ii_x, ii_y, 0]).reshape(3, 1)
                frame_coords = np.tile(
                    frame_coord,
                    (1, count),
                )
                frame_coords[2, :] = np.arange(0, count)

                coords.append(frame_coords)
                data.append(intensity_array)
                mz_data.append(mz_array)

            update(1)

        axis = np.concatenate(mz_data)
        labels = np.searchsorted(self.mz_edges[1:], axis)
        labels[labels == self.total_shape[-1]] = self.total_shape[-1] - 1
        final_coords = np.concatenate(coords, axis=1)
        final_coords[2, :] = labels

        return MultiCOO(
            coords=final_coords,
            signal=np.concatenate(data),
            axis=[axis],
        )
