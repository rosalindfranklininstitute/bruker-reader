# Bruker Reader

A simple tool for reading Bruker TIMS TOFF data and putting it into NeXus format.
The data is stored in 1 layer, with demensions: \[x, y, inverse ion mobility, mz\]

There are options to store the data:
- inflated, by supplying a bin width. The the mz axis will be xonstant thoughout.
- as peaks (same as raw). The mz axis will be 5 dimensional, the sam ea s the data.


The easiest way to inflate a particular pixel (x, y, inv ion mobility (imm)) is to use `np.histogram`:
```python
mass_edges = [bin_width*ii + min_mz for ii in range(mass_range//bin_width+1)]
masses, _ = np.histogram(
    nxs["/entry/spectra/data/mass"][0, x, y, imm, :],
    weights=nxs["/entry/spectra/data/signal"][0, x, y, imm, :],
    bins=mass_edges,
)
```
