import argparse, cfgrib, datetime, sys, numpy as np, torch
from netCDF4 import Dataset

import config
from data.io import EcmwfInputReader, OutputConverter, detect_params
from model import Kunyu
from utils import get_latlon, rollout

parser = argparse.ArgumentParser()
parser.add_argument('date', type=str, help='Timestamp in YYmmddHH format')
parser.add_argument('input_prefix', type=str, help='Prefix of input levels and surface grib files')
parser.add_argument('output', type=str, help='Location of output netcdf file')
parser.add_argument('-m', '--model', type=str, default='kunyu.model', help='Path to the model checkpoint. Default to "kunyu.model".')
parser.add_argument('-l', '--lead-time', type=int, default=240, help='Maximum lead time of desired forecast in hours. Default to 240.')
parser.add_argument('-p', '--pres-levels', type=float, nargs='+', help='Pressure levels in hPa to convert in the output files. If not specified, atmospheric variables will be in native model levels.')
parser.add_argument('--add-q', action='store_true', help='Add specific humidity (q) into output files')
parser.add_argument('--add-mslp', action='store_true', help='Add mean sea level pressure (mslp) into output files')
parser.add_argument('--legacy', action='store_true', help='Run model in legacy mode')
args = parser.parse_args()

date = datetime.datetime.strptime(args.date, '%Y%m%d%H')
levels_ds = cfgrib.open_datasets(f'{args.input_prefix}_levels.grib')
surface_ds = cfgrib.open_dataset(f'{args.input_prefix}_surface.grib')
lmax, reduced_grid = detect_params(levels_ds)
z0 = np.load('z0.npy')

reader = EcmwfInputReader(lmax, reduced_grid, z0)
pres_levels = [x * 100 for x in args.pres_levels] if args.pres_levels is not None else None
writer = OutputConverter(z0, pres_levels=pres_levels, add_q=args.add_q, add_mslp=args.add_mslp)

inputs = reader(levels_ds, surface_ds).astype(np.float32)
inputs = torch.from_numpy(inputs)[None]
consts = torch.from_numpy(np.load('consts.npy'))

model = Kunyu(args.legacy)
model.load_state_dict(torch.load(args.model))
if torch.cuda.is_available():
    inputs = inputs.cuda()
    consts = consts.cuda()
    model = model.cuda()
steps = args.lead_time // 6

out_ds = Dataset(args.output, 'w')
lat, lon = get_latlon()
out_ds.createDimension('lead_time', steps + 1)
out_ds.createDimension('level', len(pres_levels) if pres_levels is not None else config.NLEVELS)
out_ds.createDimension('latitude', config.FULLSIZE_X)
out_ds.createDimension('longitude', config.FULLSIZE_Y)
out_ds.createVariable('lead_time', np.int32, ('lead_time',))[:] = np.arange(steps + 1) * 6
if pres_levels is not None:
    out_ds.createVariable('level', np.float32, ('level',))[:] = pres_levels
out_ds.createVariable('latitude', np.float32, ('latitude',))[:] = lat / np.pi * 180
out_ds.createVariable('longitude', np.float32, ('longitude',))[:] = lon / np.pi * 180

def dump_output(ds, data, step):
    for var in data:
        if not var in ds.variables:
            dims = ('latitude', 'longitude')
            if data[var].ndim == 3: dims = ('level',) + dims
            v = ds.createVariable(var, np.float32, ('lead_time',) + dims, fill_value=np.nan)
        else:
            v = ds.variables[var]
        var_step = step
        if args.legacy and var in config.VAR_LIST['output_only']:
            var_step -= 1
        if step >= 0:
            v[var_step] = data[var]

data = writer(inputs[0].cpu().numpy())
dump_output(out_ds, data, 0)
for step, outputs in enumerate(rollout(model, date, inputs, consts, steps)):
    data = writer(outputs[0].cpu().numpy())
    dump_output(out_ds, data, step + 1)

out_ds.close()
