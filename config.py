import json, math, os, datetime

MAIN_CHANNEL = 195
INPUT_CHANNEL = 281
OUTPUT_CHANNEL = 205
FULLSIZE_X = 512
FULLSIZE_Y = 1024
LMAX = 255
DROPOUT = 0.0
TIME_STEP = datetime.timedelta(0, 6 * 3600)

LEVELS_VARS = ['z', 't', 'r', 'u', 'v']
SURFACE_VARS = ['sp', 'skt', 'src', 'stl1', 'swvl1', 'istl1', 'sd', 'rsn', 'tsn', 'siconc']
NLEVELS = 37
VAR_LIST = {'main': [], 'output_only': ['2t', '2d', '10u', '10v', 'tcc', 'hcc', 'mcc', 'lcc', 'tp', 'sf']}
for var in LEVELS_VARS:
    if var == 'r':
        R_VARS = slice(len(VAR_LIST['main']), len(VAR_LIST['main']) + NLEVELS)
    for i in range(NLEVELS):
        VAR_LIST['main'].append(var + str(i))
VAR_LIST['main'] += SURFACE_VARS
_json_file = os.path.join(os.path.dirname(__file__), 'var_stats.json')
with open(_json_file, 'r') as fp:
    VAR_STATS = json.load(fp)
