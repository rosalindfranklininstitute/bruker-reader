<!--
SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>

SPDX-License-Identifier: Apache-2.0
-->

# Bruker Reader

A simple tool for reading Bruker TIMS TOFF data and putting it into NeXus
format. For tof data the output are two hypercubes of \[x, y, mz\]: "images"
chunked for images. "spectra": chunked for spectrum.

The tims tof data is stored with dimensions: \[x, y, inverse ion mobility,
mz\]. The chunks are cubes in the data. Note that TIMS-TOF data is typically
super sparse (>1% non-zeros). The sparse chunking does not handle this well,
therefor, the output files are typically much larger than the inputs.

