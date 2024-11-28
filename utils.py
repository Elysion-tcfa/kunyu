import datetime, numpy as np, torch

import config

def _calc_julian(ts):
    Y, m, d = ts.year, ts.month, ts.day
    date = datetime.datetime(Y, m, d)
    if m <= 2:
        Y -= 1
        m += 12
    a = Y // 100
    b = 2 - a + a // 4
    return int(365.25 * (Y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5 + (ts - date).total_seconds() / 86400.

def _calc_julian_cent(ts):
    return (_calc_julian(ts) - 2451545.) / 36525.

def _calc_obliquity(t):
    seconds = 21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813))
    e0 = 23. + (26. + seconds / 60.) / 60.
    omega = (125.04 - 1934.136 * t) / 180. * np.pi
    return (e0 + 0.00256 * np.cos(omega)) / 180. * np.pi

def _calc_sun_long_params(t):
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) / 180. * np.pi
    m = (357.52911 + t * (35999.05029 - 0.0001537 * t)) / 180. * np.pi
    c = (np.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t)) + np.sin(2. * m) * (0.019993 - 0.000101 * t) + np.sin(3. * m) * 0.000289) / 180. * np.pi
    return l0, m, c

def _calc_eccentricity(t):
    return 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

def _calc_solar_time(ts):
    t = _calc_julian_cent(ts)
    epsilon = _calc_obliquity(t)
    l0, m, c = _calc_sun_long_params(t)
    e = _calc_eccentricity(t)
    y = np.tan(epsilon / 2.) ** 2
    adjustment = y * np.sin(2. * l0) - 2. * e * np.sin(m) + 4. * e * y * np.sin(m) * np.cos(2. * l0) - 0.5 * y * y * np.sin(4. * l0) - 1.25 * e * e * np.sin(2. * m)
    return ts + datetime.timedelta(0, adjustment / np.pi * 43200.)

def _calc_sun_longitude(ts):
    t = _calc_julian_cent(ts)
    l0, m, c = _calc_sun_long_params(t)
    omega = (125.04 - 1934.136 * t) / 180. * np.pi
    l = l0 + c - (0.00569 + 0.00478 * np.sin(omega)) / 180. * np.pi
    return l

def _calc_declination(ts):
    t = _calc_julian_cent(ts)
    e = _calc_obliquity(t)
    l = _calc_sun_longitude(ts)
    return np.arcsin(np.sin(e) * np.sin(l))

def _calc_earth_sun_dist(ts):
    t = _calc_julian_cent(ts)
    l0, m, c = _calc_sun_long_params(t)
    e = _calc_eccentricity(t)
    return 1.000001018 * (1. - e * e) / (1. + e * np.cos(m + c))

def _calc_zenith_azimuth(theta, lats, hour_angle):
    zenith = np.sin(lats) * np.sin(theta) + np.cos(lats) * np.cos(theta) * np.cos(hour_angle)
    azimuth_x = np.where(np.abs(np.cos(lats)) < 1e-3, 0., (np.sin(theta) - np.sin(lats) * zenith) / np.cos(lats))
    azimuth_y = np.where(np.abs(np.cos(lats)) < 1e-3, 0., np.sqrt(np.maximum(1. - zenith ** 2 - azimuth_x ** 2, 0.)))
    azimuth_y = np.where(np.sin(hour_angle) > 0, -azimuth_y, azimuth_y)
    return [zenith, azimuth_x * np.cos(lats), azimuth_y * np.cos(lats)]

def get_latlon():
    lats = np.arange(config.FULLSIZE_X - 1, -config.FULLSIZE_X - 1, -2) / (2 * config.FULLSIZE_X) * np.pi
    lons = np.arange(config.FULLSIZE_Y) / config.FULLSIZE_Y * (2 * np.pi)
    return lats, lons

def get_time_features(ts):
    lats, lons = get_latlon()
    lats, lons = np.meshgrid(lats.astype(np.float32), lons.astype(np.float32), indexing='ij')
    solar_time = _calc_solar_time(ts)
    solar_time = (solar_time - datetime.datetime(solar_time.year, solar_time.month, solar_time.day)).total_seconds() / 43200. * np.pi
    hour_angle = solar_time + lons - np.pi
    sun_lon = _calc_sun_longitude(ts)
    theta = _calc_declination(ts)
    zenith_azimuth1 = _calc_zenith_azimuth(theta, lats, hour_angle)
    zenith_azimuth2 = _calc_zenith_azimuth(theta, lats, hour_angle + np.pi / 12.)
    ones = np.ones_like(lats, dtype=np.float32)
    const_inputs = list(map(lambda x: ones * x, (np.cos(sun_lon), np.sin(sun_lon), np.cos(theta), np.sin(theta), (_calc_earth_sun_dist(ts) - 1.) / 0.0118)))
    lon_inputs = [np.cos(hour_angle) * np.cos(lats), np.sin(hour_angle) * np.cos(lats)] + zenith_azimuth1 + zenith_azimuth2
    return np.stack(const_inputs + lon_inputs)

def rollout(model, ts, inputs, consts, steps):
    consts = consts[None].repeat(inputs.shape[0],1,1,1)
    for _ in range(steps):
        time_features = torch.from_numpy(get_time_features(ts)[None]).to(inputs.device).repeat(inputs.shape[0],1,1,1)
        inputs = torch.cat([inputs, time_features, consts], dim=1)
        with torch.no_grad():
            with torch.autocast('cuda', torch.float16):
                inputs = model(inputs)
        yield inputs
        inputs = inputs[:, :config.MAIN_CHANNEL]
        ts += config.TIME_STEP
