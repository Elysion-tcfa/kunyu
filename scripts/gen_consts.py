import cdsapi, cfgrib, numpy as np

import config
from data.regrid import ReducedGrid, Regridder

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-complete',
    {
        'date': '2019-01-01',
        'time': '00',
        'levtype': 'sfc',
        'param': '27/28/29/30/43/129/160/161/162/163/172',
        'stream': 'oper',
        'type': 'an',
        'format': 'grib',
    },
    'consts.grib')

lats = np.pi / 2 - (np.array(list(range(config.FULLSIZE_X)), dtype='float32') + 0.5) / config.FULLSIZE_X * np.pi
lons = np.array(list(range(config.FULLSIZE_Y)), dtype='float32') / config.FULLSIZE_Y * 2 * np.pi
ds = cfgrib.open_dataset('consts.grib')
regridder = Regridder(ReducedGrid.from_xarray_ds(ds))

const_values = []
lat_value = np.repeat(np.reshape(lats, (-1, 1)), lons.shape[0], axis=1)
const_values.append(np.cos(lat_value).astype('float32'))
const_values.append(np.sin(lat_value).astype('float32'))
for var in ['z', 'sdor', 'isor', 'slor', 'anor', 'lsm', 'cvl', 'cvh', 'slt']:
    data_input = ds.variables[var].data
    if var == 'sdor': sdor = data_input
    if var == 'cvl':
        tvl = ds.variables['tvl'].data
        values = [data_input * (np.abs(tvl - i) < 1e-3) for i in range(20)]
    elif var == 'cvh':
        tvh = ds.variables['tvh'].data
        values = [data_input * (np.abs(tvh - i) < 1e-3) for i in range(20)]
    elif var == 'slt':
        values = [(np.abs(data_input - i) < 1e-3).astype(np.float32) for i in range(8)]
    elif var == 'anor':
        values = [np.cos(data_input) * sdor, np.sin(data_input) * sdor]
    elif var == 'isor':
        values = [data_input * sdor]
    else:
        values = [data_input]
    for value in values:
        value = regridder(value)
        if var == 'z': z0 = value
        if var == 'anor': value *= np.cos(lat_value)
        if var in ['z', 'sdor', 'slor', 'isor', 'anor']:
            value = value / np.sqrt((value ** 2).mean())
        const_values.append(value.astype(np.float32))
const_input = np.stack(const_values, axis=0)
np.save('consts.npy', const_input.data)
np.save('z0.npy', z0.data)
