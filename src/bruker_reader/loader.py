from pathlib import Path
import os
from contextlib import contextmanager


@contextmanager
def load_opentims(path):

    from ctypes import util

    tmp_find_library = util.find_library

    def intercept_find_library(name: str) -> str | None:
        if name == "sqlite3":
            return str(Path(os.path.dirname(__file__)) / "dlls" / "sqlite3.dll")
        else:
            return tmp_find_library(name)

    util.find_library = intercept_find_library

    try:
        from opentimspy import OpenTIMS
        import opentimspy

        if not opentimspy.bruker_bridge_present:
            opentimspy.setup_opensource()

        with OpenTIMS(path) as D:
            yield D

    finally:
        util.find_library = tmp_find_library
