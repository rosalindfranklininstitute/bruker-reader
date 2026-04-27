from dataclasses import dataclass

from ms_nexus_tools.lib.bounds import Shape
from ms_nexus_tools.api.mass_range_args import MassRange
from datargs.args import arg_field

import numpy as np


@dataclass
class MobilityCentreArgs:
    mass_centre: list[str] = arg_field(
        action="append",
        doc="Each instance of this adds a plot of the total spectra centered around this mass or formula, with width --mass-width",
        default_factory=list,
    )

    mass_width: float = arg_field(
        doc="The mass range to plot around each mass centre.",
        default=1.0,
    )

    def calculate_centre_ranges(self, mass_values: np.ndarray) -> list[MassRange]:

        return [
            MassRange.from_centre_and_width(f, self.mass_width, mass_values)
            for f in self.mass_centre
        ]
