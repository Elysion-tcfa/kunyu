import numpy as np

import config
from utils import get_latlon

class ReducedGrid:
    def __init__(self, lats, lat_points):
        self.lats = lats
        self.lat_points = lat_points

    @staticmethod
    def from_xarray_ds(ds):
        n320_lat, n320_lon = ds.latitude.data, ds.longitude.data
        lats, lat_points = [], []
        prev = 0
        for i in range(1, len(n320_lat)+1):
            if i == len(n320_lat) or n320_lat[i] != n320_lat[i-1]:
                lats.append(n320_lat[i-1] / 180 * np.pi)
                lat_points.append(i - prev)
                prev = i
        return ReducedGrid(lats, lat_points)

class Regridder:
    def __init__(self, reduced_grid):
        self.reduced_grid = reduced_grid

        lon_indices1, lon_indices2, lon_weights = [], [], []
        to_lons = np.arange(config.FULLSIZE_Y) / config.FULLSIZE_Y
        cnt = 0
        for points in reduced_grid.lat_points:
            x = to_lons * points
            indices = np.floor(x).astype(np.int64)
            lon_indices1.append(indices + cnt)
            lon_indices2.append((indices + 1) % points + cnt)
            lon_weights.append(1 - x + indices)
            cnt += points
        self.lon_indices = np.stack([np.concatenate(lon_indices1, axis=0), np.concatenate(lon_indices2, axis=0)], axis=0)
        lon_weights = np.concatenate(lon_weights, axis=0)
        self.lon_weights = np.stack([lon_weights, 1 - lon_weights], axis=0)

        from_lats = np.concatenate([np.array([np.pi/2]), np.array(reduced_grid.lats), np.array([-np.pi/2])], axis=0)
        to_lats = get_latlon()[0]
        lat_indices, lat_weights = [], []
        i = 0
        for j in range(len(to_lats)):
            while i+1 < len(from_lats) and from_lats[i+1] > to_lats[j]: i += 1
            assert from_lats[i] >= to_lats[j] and from_lats[i+1] <= to_lats[j]
            lat_indices.append(i)
            lat_weights.append((to_lats[j]-from_lats[i+1]) / (from_lats[i]-from_lats[i+1]))
        lat_indices, lat_weights = [np.array(x) for x in [lat_indices, lat_weights]]
        self.lat_indices = np.stack([lat_indices, lat_indices+1], axis=0)
        self.lat_weights = np.stack([lat_weights, 1-lat_weights], axis=0)

    def __call__(self, in_data):
        lon_interp_data = in_data[...,self.lon_indices[0]] * self.lon_weights[0] + \
                in_data[...,self.lon_indices[1]] * self.lon_weights[1]
        npole = in_data[...,:self.reduced_grid.lat_points[0]].mean(axis=-1, keepdims=True)
        spole = in_data[...,-self.reduced_grid.lat_points[-1]:].mean(axis=-1, keepdims=True)
        lon_interp_data = lon_interp_data.reshape(lon_interp_data.shape[:-1] + (len(self.reduced_grid.lats), config.FULLSIZE_Y))
        lon_interp_data = np.concatenate([np.ones_like(lon_interp_data[...,0:1,:]) * npole[...,None,:],
            lon_interp_data,
            np.ones_like(lon_interp_data[...,-1:,:]) * spole[...,None,:]], axis=-2)
        return lon_interp_data[...,self.lat_indices[0],:] * self.lat_weights[0,:,None] + \
                lon_interp_data[...,self.lat_indices[1],:] * self.lat_weights[1,:,None]
