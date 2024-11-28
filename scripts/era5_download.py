import cdsapi, sys, datetime, os

if len(sys.argv) < 3:
    print('usage:', os.path.basename(sys.argv[0]), '<date in YYmmddHH> <output_prefix>')
    quit(2)

date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d%H')
fn = sys.argv[2]

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-complete',
    {
        'date': date.strftime('%Y-%m-%d'),
        'time': '%02d' % date.hour,
        'levelist': '/'.join(map(str, range(1, 138))),
        'levtype': 'ml',
        'param': '75/76/129/130/131/132/133/246/247',
        'stream': 'oper',
        'type': 'an',
        'format': 'grib'
    }, f'{fn}_levels.grib')
c.retrieve(
    'reanalysis-era5-complete',
    {
        'date': date.strftime('%Y-%m-%d'),
        'time': '%02d' % date.hour,
        'levtype': 'sfc',
        'param': '31/33/35/39/134/139/141/198/235/238',
        'stream': 'oper',
        'type': 'an',
        'format': 'grib',
    }, f'{fn}_surface.grib')
