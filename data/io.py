import numpy as np, torch, torch_sht

import config
from data.regrid import ReducedGrid, Regridder
from utils import get_latlon

L137_A = np.array([0, 2.000365, 3.102241, 4.666084, 6.827977, 9.746966, 13.605424, 18.608931, 24.985718, 32.98571, 42.879242, 54.955463, 69.520576, 86.895882, 107.415741, 131.425507, 159.279404, 191.338562, 227.968948, 269.539581, 316.420746, 368.982361, 427.592499, 492.616028, 564.413452, 643.339905, 729.744141, 823.967834, 926.34491, 1037.201172, 1156.853638, 1285.610352, 1423.770142, 1571.622925, 1729.448975, 1897.519287, 2076.095947, 2265.431641, 2465.770508, 2677.348145, 2900.391357, 3135.119385, 3381.743652, 3640.468262, 3911.490479, 4194.930664, 4490.817383, 4799.149414, 5119.89502, 5452.990723, 5798.344727, 6156.074219, 6526.946777, 6911.870605, 7311.869141, 7727.412109, 8159.354004, 8608.525391, 9076.400391, 9562.682617, 10065.978516, 10584.631836, 11116.662109, 11660.067383, 12211.547852, 12766.873047, 13324.668945, 13881.331055, 14432.139648, 14975.615234, 15508.256836, 16026.115234, 16527.322266, 17008.789063, 17467.613281, 17901.621094, 18308.433594, 18685.71875, 19031.289063, 19343.511719, 19620.042969, 19859.390625, 20059.931641, 20219.664063, 20337.863281, 20412.308594, 20442.078125, 20425.71875, 20361.816406, 20249.511719, 20087.085938, 19874.025391, 19608.572266, 19290.226563, 18917.460938, 18489.707031, 18006.925781, 17471.839844, 16888.6875, 16262.046875, 15596.695313, 14898.453125, 14173.324219, 13427.769531, 12668.257813, 11901.339844, 11133.304688, 10370.175781, 9617.515625, 8880.453125, 8163.375, 7470.34375, 6804.421875, 6168.53125, 5564.382813, 4993.796875, 4457.375, 3955.960938, 3489.234375, 3057.265625, 2659.140625, 2294.242188, 1961.5, 1659.476563, 1387.546875, 1143.25, 926.507813, 734.992188, 568.0625, 424.414063, 302.476563, 202.484375, 122.101563, 62.78125, 22.835938, 3.757813, 0, 0])
L137_B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000007, 0.000024, 0.000059, 0.000112, 0.000199, 0.00034, 0.000562, 0.00089, 0.001353, 0.001992, 0.002857, 0.003971, 0.005378, 0.007133, 0.009261, 0.011806, 0.014816, 0.018318, 0.022355, 0.026964, 0.032176, 0.038026, 0.044548, 0.051773, 0.059728, 0.068448, 0.077958, 0.088286, 0.099462, 0.111505, 0.124448, 0.138313, 0.153125, 0.16891, 0.185689, 0.203491, 0.222333, 0.242244, 0.263242, 0.285354, 0.308598, 0.332939, 0.358254, 0.384363, 0.411125, 0.438391, 0.466003, 0.4938, 0.521619, 0.549301, 0.576692, 0.603648, 0.630036, 0.655736, 0.680643, 0.704669, 0.727739, 0.749797, 0.770798, 0.790717, 0.809536, 0.827256, 0.843881, 0.859432, 0.873929, 0.887408, 0.8999, 0.911448, 0.922096, 0.931881, 0.94086, 0.949064, 0.95655, 0.963352, 0.969513, 0.975078, 0.980072, 0.984542, 0.9885, 0.991984, 0.995003, 0.99763, 1])
SELECT_LAYERS = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134][::-1])
SELECT_A = (L137_A[SELECT_LAYERS] + L137_A[SELECT_LAYERS + 1]) / 2
SELECT_B = (L137_B[SELECT_LAYERS] + L137_B[SELECT_LAYERS + 1]) / 2
EXTRAPL_LVL = 1

class ReducedSHT:
    def __init__(self, lmax, reduced_grid):
        self.reduced_grid = reduced_grid
        self.sht = torch_sht.SHT(lmax, norm=1|1024)
        self.sht.set_grid(len(reduced_grid.lats), 2 * (lmax + 1))

    def sh_to_grid(self, in_data):
        def ifft_num(tensor, num):
            m = tensor.shape[-1]
            num1 = (m + num - 1) // num * num
            if num1 % 2 == 1: num1 += num
            pad = (num1 - m) // 2
            ret = torch.fft.ifft(torch.cat([tensor[...,:m//2], torch.zeros_like(tensor[...,:pad]), tensor[...,m//2:m//2+1], torch.zeros_like(tensor[...,:pad]), tensor[...,m//2+1:]], dim=-1))
            ret = (ret.real * (num1 / m))[...,::num1//num]
            return ret
        out_fft = torch.fft.fft(self.sht.sh_to_grid(in_data), dim=-2)
        out_data = torch.cat([ifft_num(out_fft[...,i], num) for i, num in enumerate(self.reduced_grid.lat_points)], dim=-1)
        return out_data

class EcmwfInputReader:
    def __init__(self, lmax, reduced_grid, z0):
        self.regridder = Regridder(reduced_grid)
        self.sht = ReducedSHT(lmax, reduced_grid)
        self.z0 = z0

    def _get_data(self, ds_list, var, slicing=()):
        for ds in ds_list:
            if var in ds:
                ret = ds.variables[var].data[slicing]
                if ds.variables[var].attrs['GRIB_gridType'] == 'sh':
                    ret = torch.from_numpy(ret)
                    ret = torch.view_as_complex(ret.view(ret.shape[:-1] + (-1, 2)))
                    ret = self.sht.sh_to_grid(ret).numpy()
                return ret

    def _calc_z(self, z_h, t, q, sp, cwc):
        z = np.zeros(t.shape)
        p = L137_A[137] + L137_B[137] * sp
        for lev in range(136, -1, -1):
            pn = L137_A[lev] + L137_B[lev] * sp
            if lev == 0:
                alpha = np.log(2)
            else:
                dlogp = np.log(p / pn)
                alpha = 1. - pn / (p - pn) * dlogp
            tv = t[lev] * (1. + 0.609133 * q[lev] - cwc[lev]) * 287.06
            z[lev] = z_h + tv * alpha
            if lev > 0: z_h += tv * dlogp
            p = pn
        return z

    def __call__(self, levels_ds_list, surface_ds):
        cwc = self._get_data(levels_ds_list, 'clwc')
        for var in ['crwc', 'ciwc', 'cswc']:
            cwc += self._get_data(levels_ds_list, var)
        t, q = [self._get_data(levels_ds_list, var) for var in ['t', 'q']]
        sp = surface_ds.variables['sp'].data
        z = self._calc_z(self._get_data(levels_ds_list, 'z'), t, q, sp, cwc)
        del cwc

        z, t, q = [self.regridder(tensor[SELECT_LAYERS]) for tensor in [z, t, q]]
        sp = self.regridder(sp)
        z -= self.z0[None] * SELECT_B[:,None,None]
        e_sat = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))
        p = SELECT_A[:,None,None] + SELECT_B[:,None,None] * sp
        r = 1.609133 * p * q / (e_sat * (1 + 0.609133 * q))
        tensors = {'z': z, 't': t, 'r': r}
        del e_sat, p, q, r, sp, t, z

        for var in ['u', 'v']:
            tensors[var] = self.regridder(self._get_data(levels_ds_list, var, SELECT_LAYERS))
        ret = {}
        for var in config.LEVELS_VARS:
            for lvl in range(config.NLEVELS):
                v = var + str(lvl)
                ret[v] = (tensors[var][lvl] - config.VAR_STATS[v]['mean']) / config.VAR_STATS[v]['std']
        del tensors

        sd = surface_ds.variables['sd'].data
        siconc = surface_ds.variables['siconc'].data
        siconc = np.where(np.isnan(siconc), 0., siconc)
        for var in config.SURFACE_VARS:
            val = surface_ds.variables[var].data
            val = np.where(np.isnan(val), 0., val)
            if var == 'tsn':
                val = np.where(sd + siconc > 1e-4, val, 273.16)
            if var == 'istl1':
                val = np.where(siconc > 1e-4, val, 273.16)
            ret[var] = (self.regridder(val) - config.VAR_STATS[var]['mean']) / config.VAR_STATS[var]['std']

        return np.stack([ret[var] for var in config.VAR_LIST['main']], axis=0)

class OutputConverter:
    def __init__(self, z0, pres_levels=None, add_q=False, add_mslp=False):
        self.z0 = z0
        self.convert_pl = bool(pres_levels)
        if self.convert_pl:
            self.levels = pres_levels
            self.cos_lat = np.cos(get_latlon()[0])[:,None]
        self.add_q = add_q
        self.add_mslp = add_mslp

    def _convert_raw(self, raw):
        var_data = {}
        for idx, var in enumerate(config.VAR_LIST['main'] + config.VAR_LIST['output_only']):
            if idx < raw.shape[0]:
                var_data[var] = raw[idx] * config.VAR_STATS[var]['std'] + config.VAR_STATS[var]['mean']
        levels_data = {}
        for var in config.LEVELS_VARS:
            levels_data[var] = np.stack([var_data[var + str(lvl)] for lvl in range(config.NLEVELS)])
        levels_data['z'] += self.z0[None] * SELECT_B[:,None,None]

        if self.add_q:
            t = levels_data['t']
            e = levels_data['r'] * 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))
            p = SELECT_A[:,None,None] + SELECT_B[:,None,None] * var_data['sp']
            levels_data['q'] = e / (1.609133 * p - 0.609133 * e)
        return levels_data, {k: v for k, v in var_data.items() if k in config.SURFACE_VARS + config.VAR_LIST['output_only']}

    def _ml_to_pl(self, in_data, var, t, z0, sp, lvl_p):
        if var != 'mslp':
            spatial_shape = in_data.shape[1:]
            in_data = in_data.reshape((in_data.shape[0], -1))
            sort_idx = np.argsort(sp.flatten())
            sorted_sp = sp.flatten()[sort_idx]
            out_data = np.full((len(lvl_p), sorted_sp.shape[0]), np.nan, dtype=in_data.dtype)
            for i in range(config.NLEVELS - 1):
                for j, p in enumerate(lvl_p):
                    sp_l = (p - SELECT_A[i]) / max(SELECT_B[i], 1e-12)
                    sp_r = (p - SELECT_A[i + 1]) / max(SELECT_B[i + 1], 1e-12)
                    l = np.searchsorted(sorted_sp, sp_l)
                    r = np.searchsorted(sorted_sp, sp_r, side='right')
                    ml_p1 = SELECT_A[i] + SELECT_B[i] * sorted_sp[l:r]
                    ml_p2 = SELECT_A[i + 1] + SELECT_B[i + 1] * sorted_sp[l:r]
                    if var == 'z':
                        p, ml_p1, ml_p2 = map(np.log, (p, ml_p1, ml_p2))
                    out_data[j, sort_idx[l:r]] = in_data[i, sort_idx[l:r]] + (p - ml_p1) / (ml_p2 - ml_p1) * (in_data[i+1, sort_idx[l:r]] - in_data[i, sort_idx[l:r]])
            in_data = in_data.reshape((-1,) + spatial_shape)
            out_data = out_data.reshape((-1,) + spatial_shape)

        if var in ['t', 'z', 'mslp']:
            sel_ml_p = SELECT_A[EXTRAPL_LVL] + SELECT_B[EXTRAPL_LVL] * sp
            t_surf = t[EXTRAPL_LVL] * (1 + 0.0065 * 287.06 / 9.807 * (sp / sel_ml_p - 1))
            h0 = z0 / 9.807
            t0 = t_surf + 0.0065 * h0
            if var == 't':
                t0_c = np.clip(t0, None, 298.)
                gamma = np.where(h0 < 2000., np.full_like(h0, 0.0065), np.clip(np.where(h0 < 2500., t0 + (h0 - 2000.) / 500. * (t0_c - t0), t0_c) - t_surf, 0., None) / h0)
            else:
                t_surf = np.where(t_surf < 255, 0.5 * (255 + t_surf), t_surf)
                t_surf = np.where((t0 > 290.5) & (t_surf > 290.5), 0.5 * (290.5 + t_surf), t_surf)
                gamma = np.where(t0 > 290.5, np.clip(290.5 - t_surf, 0.001, None) * 9.807 / z0, np.full_like(t0, 0.0065))
            if var == 'mslp':
                y = gamma * (287.06 / 9.807)
                x = z0 * gamma / (9.807 * t_surf)
                out_data = sp * np.exp(np.log1p(x) / y)
            else:
                ml_p = SELECT_A[0] + SELECT_B[0] * sp
                if var == 'z':
                    ml_p = np.log(ml_p)
                    lnsp = np.log(sp)
                for j, p in enumerate(lvl_p):
                    y = gamma * (287.06 / 9.807) * (np.log(p / sp))
                    if var == 't':
                        out_data[j] = np.where((ml_p <= p) & (p <= sp), t_surf + (sp - p) / (sp - ml_p) * (in_data[0] - t_surf), out_data[j])
                        out_data[j] = np.where(p > sp, t_surf * np.exp(y), out_data[j])
                    else: # z
                        p = np.log(p)
                        out_data[j] = np.where((ml_p <= p) & (p <= lnsp), z0 + (lnsp - p) / (lnsp - ml_p) * (in_data[0] - z0), out_data[j])
                        out_data[j] = np.where(p <= lnsp, out_data[j], z0 - t_surf * (9.807 / gamma) * np.expm1(y))
        else:
            ml_p = SELECT_A[0] + SELECT_B[0] * sp
            for j, p in enumerate(lvl_p):
                out_data[j] = np.where(ml_p < p, in_data[0], out_data[j])

        return out_data

    def __call__(self, raw):
        levels_data, surface_data = self._convert_raw(raw)

        t = levels_data['t']
        if self.convert_pl:
            for var in levels_data:
                levels_data[var] = self._ml_to_pl(levels_data[var], var, t, self.z0, surface_data['sp'], self.levels)
                if var in ['u', 'v']: levels_data[var] /= self.cos_lat
        if self.add_mslp:
            surface_data['mslp'] = self._ml_to_pl(None, 'mslp', t, self.z0, surface_data['sp'], None)

        levels_data.update(surface_data)
        return levels_data

def detect_params(levels_ds):
    lmax, reduced_grid = None, None
    for ds in levels_ds:
        if 'q' in ds:
            reduced_grid = ReducedGrid.from_xarray_ds(ds)
        elif 't' in ds:
            lmax = ds.variables['t'].attrs['GRIB_M']
    return lmax, reduced_grid
