from dataclasses import dataclass, field
import numpy as np

Float1D32 = np.ndarray[tuple[int], np.dtype[np.float32]]
Float2D32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
Int1D32 = np.ndarray[tuple[int], np.dtype[np.int32]]
Int2D32 = np.ndarray[tuple[int, int], np.dtype[np.int32]]


@dataclass(init=False)
class MaldiAxis:
    shape: tuple[int, int]
    frame_x: Int1D32
    frame_y: Int1D32
    frame_x_to_axis: dict[int, int]
    frame_y_to_axis: dict[int, int]
    frame_inx: Int2D32

    def __init__(self, maldi_frame_info):
        frame_id = maldi_frame_info["Frame"]
        self.frame_x = maldi_frame_info["XIndexPos"]
        self.frame_y = maldi_frame_info["YIndexPos"]

        self.x_values = np.unique(self.frame_x, sorted=True)
        self.y_values = np.unique(self.frame_y, sorted=True)

        self.shape = (len(self.x_values), len(self.y_values))

        self.frame_x_to_axis = {fx: ii for ii, fx in enumerate(self.x_values)}
        self.frame_y_to_axis = {fy: ii for ii, fy in enumerate(self.y_values)}

        self.frame_inx = np.full(self.shape, -1)
        for ii, x, y in zip(frame_id, self.frame_x, self.frame_y):
            ii_x = self.frame_x_to_axis[x]
            ii_y = self.frame_y_to_axis[y]
            assert self.frame_inx[ii_x, ii_y] == -1
            self.frame_inx[ii_x, ii_y] = ii
