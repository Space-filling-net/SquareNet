import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def distfunction(x, y, axis):
    return ((x-y)**2).sum(axis = axis)

def distkernel(x, xview):
    d = len(x.shape) - 1
    return distfunction(x[..., None, None], xview, axis = d)

def heatmap(grided_points, criterion = "rank", thresholdcut = 1, kernel = distkernel, windowradius= 5, gridaxes = (0,1), max_sample_size = 100_000):
    gpts = grided_points.copy()
    d = len(gpts.shape) - 1
    wr = windowradius
    ws = 2*wr+1

    shp = np.array(gpts.shape, dtype = int)[:-1]
    if np.prod(shp) > max_sample_size:
        targetshp = int(max_sample_size**(1/d))
        shp = shp.clip(max = targetshp)
    slices = tuple(slice(0, s) for s in shp) + (slice(None),)
    gpts = gpts[slices]

        
    pad_width = [(0, 0)]*(d+1)
    for gax in gridaxes:
        pad_width[gax] = (wr, wr)
    gpad = np.pad(
        gpts,
        pad_width=pad_width,
        mode="constant",
        constant_values = np.nan
    )

    gview = sliding_window_view(
        gpad,
        window_shape=(ws, ws),
        axis=gridaxes
    )


    dists = kernel(gpts, gview)

    if criterion =="value":
        hotspots = (dists <= thresholdcut).sum(axis = tuple(range(d)))
    if criterion == "rank":
        dists = dists.reshape(-1, ws*ws)
        ranks = np.argsort(np.argsort(dists, axis = -1), axis = -1) - 1
        hotspots = (ranks <= thresholdcut).sum(axis = 0).reshape(ws, ws)
    hotspots[wr, wr] = 0
    return  hotspots