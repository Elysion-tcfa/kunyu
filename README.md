# Kunyu: A High-Performing Global Weather Model Beyond Regression Losses

This repository contains implementation of the Kunyu and Kunyu-Legacy models introduced in this [arXiv paper](https://arxiv.org/abs/2312.08264), as well as example scripts to inference the models using data from ERA5.

## Installation

In order to run Kunyu, three basic packages should be installed first: `numpy`,  `scipy` and  `torch`.

In order to go through the example below, you are also required to install the following four additional packages for data processing purposes: `cdsapi`, `cfgrib`, `netcdf4` and `xarray`.

### Install SHTns

Additionally, Kunyu requires the SHTns package as well as a custom PyTorch binding to carry out spherical harmonics transforms.

- First, install FFTW and PyBind11. In Ubuntu and Debian systems, use the following command
  ```bash
  sudo apt-get install libfftw3-dev python3-pybind11
  ```
- Second, go to https://bitbucket.org/nschaeff/shtns/downloads/ and download the latest version of SHTns (or version 3.7.3 at the time of writing).
- Under the unzipped folder, type
  ```bash
  ./configure --enable-openmp --enable-shared
  ```
  to configure SHTns for a CPU-only environment, or type
  ```bash
  ./configure --enable-openmp --enable-cuda --enable-shared
  ```
  for a GPU-enabled environment. Note that `CUDA_PATH` may need to be provided if `configure` fails to find CUDA.
- Finish installing SHTns by typing
  ```bash
  make && sudo make install
  ```
- Switch to the Kunyu repository. Under `sht` folder, install the custom `torch_sht` package by typing
  ```bash
  python3 setup.py install
  ```
  for CPU-only environments, or
  ```bash
  USE_GPU=1 python3 setup.py install
  ```
  for GPU-enabled environments.

### Download static files and model weights

- Static files can either be downloaded from Kunyu's server:
  ```bash
  wget https://kunyu.q-weather.info/media/models/z0.npy
  wget https://kunyu.q-weather.info/media/models/consts.npy
  ```
  or generated from ERA5 by executing the script under the root folder of this repository:
  ```bash
  python3 scripts/gen_consts.py
  ```
  assuming you have properly configured `cdsapi` with your CDS API key.
- Weights of the model Kunyu shall be downloaded from Kunyu's server:
  ```bash
  wget https://kunyu.q-weather.info/media/models/kunyu.model.gz && gunzip kunyu.model.gz
  ```
  or, if you'd like to run the Kunyu-Legacy model instead, type
  ```bash
  wget https://kunyu.q-weather.info/media/models/kunyu_legacy.model.gz && gunzip kunyu_legacy.model.gz
  ```
  Note that it will incur a ~7GB download.

## ERA5 Inference Example

In this example, you are assumed to have completed all steps above and downloaded static files and model weights under the root folder of this repository.

First, download initial condition from ERA5 archive. You should have properly configured `cdsapi` with your CDS API key. Kunyu requires data both from L137 model levels and in its original reduced Gaussian grid representation. Use the following script to download ERA5 data at 0 UTC on July 6, 2018:
```bash
python3 scripts/era5_download.py 2018070600 era5_2018070600
```
Here, the first argument is the desired initial timestamp in `YYmmddHH` format, and the second argument is the prefix of the downloaded files. The script will initiate two requests, one for model levels and another for surface level, and download the data to `era5_2018070600_levels.grib` and `era5_2018070600_surface.grib`, respectively.

Next, type the following command to run the Kunyu model and interpolate the output to 500, 700 and 850 hPa pressure levels:
```bash
python3 scripts/era5_infer.py 2018070600 era5_2018070600 output.nc -p 500 700 850 --add-q --add-mslp
```
In this command, the first two arguments bear the same meaning as in `era5_download.py`, and the third argument is the path to the output NetCDF file. Additional optional arguments modify the format of the output. More information can be found by typing `python3 scripts/era5_infer.py -h`. If you want to run the Kunyu-Legacy model instead, append `--legacy -m kunyu_legacy.model` to the command line. This process will generate a ~3GB output file `output.nc`.

Finally, you can do what you want with the inference results. For example, plotting `q[12,1]` should produce the same result as displayed in Figure 2 of the arXiv paper over the specified region.

## Hardware requirements

Kunyu should require at least 16GB CPU memory and, in the case of GPU inference, 16GB GPU memory. The code in this repository undergoes some optimizations to reduce inference-time memory footprint. Under 16GB GPU memory however, the flag `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is likely necessary.
