from typing import Any, Callable
from pathlib import Path

import numpy as np

import ms_nexus_tools.lib.data_source as ds
from ms_nexus_tools.lib.bounds import Shape, Chunk


class MSIDataSource(ds.AbstractDataSource):
    def __init__(self, filename: Path):
        self.inflate_mz = False
        self.D = None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def instrament_metadata(self) -> dict[str, Any]:
        assert self.D is not None
        return dict(
            mz_spectra_type=ds.AxisDensity.CONTINUOUS
            if self.inflate_mz
            else ds.AxisDensity.SPARSE,
            min_mz=self.D.min_mz,
            max_mz=self.D.max_mz,
            min_inv_ion_mobility=self.D.min_inv_ion_mobility,
            max_inv_ion_mobility=self.D.max_inv_ion_mobility,
            min_x=self.D.GlobalMetadata["ImagingAreaMinXIndexPos"],
            max_x=self.D.GlobalMetadata["ImagingAreaMaxXIndexPos"],
            min_y=self.D.GlobalMetadata["ImagingAreaMinYIndexPos"],
            max_y=self.D.GlobalMetadata["ImagingAreaMaxYIndexPos"],
        )

    def experiment_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the experiment metadata.
        """
        return dict()

    def shape(self) -> Shape:
        """
        Return the shape of the data.
        """
        return ()

    def output_chunks(self, max_items_per_chunk: int) -> dict[str, Shape]:
        """
        Returns the names and chunking priorities of the desired output array.
        For examlpe simple image data (x,y, spectra) with shape (32,32,184000)
        might produce:
        'images':   (1,1,2) -> (32,32,1)
        'spectra':  (2,2,1) -> (1,1,184000)
        """
        return dict()

    def output_accumulations(self) -> dict[str, tuple[int, ...]]:
        """
        Returns the names and lists of axis that should have
        the be accumulated (summed and max) and stored.
        For examlpe simple image data (x,y, spectra):
        might produce:
        'total_images':     (2) # Accumulate over the spectra
        'total_spectra':    (0,1) # Accumulate over the images
        """
        return dict()

    def chunk_read_count(self, memory_chunk: Chunk) -> int:
        """
        Returns the number of read operations needed to fill the provided memory chunk.
        """
        return 0

    def axis_definitions(self, index: int) -> list[ds.Axis]:
        """
        Returns the axis that should be used when storing the data.
        For examlpe simple image data (x,y, spectra):
        >>> axis(0) -> Axis('x', (len(x),), 'um')
        >>> axis(1) -> Axis('y', (len(y),), 'um')
        If is it continuous:
        >>> axis(2) -> Axis('mz', (len(spectra),))
        if it is only peaks:
        >>> axis(2) -> Axis('mz', (len(x), len(y), len(spectra),))
        """
        return []

    def axis_values(self, axis: ds.Axis) -> np.ndarray:
        """
        Returns the values for the specified axis.
        This will only be called for continuouse axis.
        """
        return np.array([])

    def fill_chunk(
        self,
        memory_chunk: Chunk,
        fill_axis: list[ds.Axis],
        update: Callable[[int], None],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Read data from the source in the region specified by
        memory_chunk and return that data. Also return the data
        any sparse axis.

        Parameters:
        memory_chunk:   The bounds of the data to read.
        fill_axis:      The list of sparce axis to fill.
        update:         A callback to update progress.
                        The total of the progress counter is
                        sum([chunk_read_count(mc) for mc in all_memory_chunks])
        Returns:
        The data from the source, and the data for all the sparse axis.

        """
        return (np.ndarray([]), [])
