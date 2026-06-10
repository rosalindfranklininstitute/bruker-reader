from typing import Callable
from pathlib import Path
import os
from contextlib import contextmanager, AbstractContextManager


class LoadOpenTims(AbstractContextManager):
    def __init__(self, path: Path):
        self.path = path

        self.tmp_find_library: Callable
        self.D: "OpenTIMS"

    def __enter__(self) -> "OpenTIMS":
        from ctypes import util

        self.tmp_find_library = util.find_library

        def intercept_find_library(name: str) -> str | None:
            if name == "sqlite3":
                return str(Path(os.path.dirname(__file__)) / "dlls" / "sqlite3.dll")
            else:
                return self.tmp_find_library(name)

        util.find_library = intercept_find_library
        try:
            from opentimspy import OpenTIMS
            import opentimspy

            if not opentimspy.bruker_bridge_present:
                opentimspy.setup_opensource()

            self.D = OpenTIMS(self.path)
            return self.D

        finally:
            util.find_library = self.tmp_find_library

    def __exit__(self, exc_type, exc_value, traceback):
        from ctypes import util

        util.find_library = self.tmp_find_library

        if self.D is not None:
            self.D.close()
