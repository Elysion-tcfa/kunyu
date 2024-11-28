import os
from setuptools import setup
from torch.utils import cpp_extension

use_gpu = os.environ.get('USE_GPU', None)

if use_gpu:
    setup(name='torch_sht', ext_modules=[
        cpp_extension.CUDAExtension('torch_sht', ['sht.cpp'],
            libraries=['shtns_cuda_omp', 'fftw3', 'fftw3_threads', 'nvrtc', 'cuda'],
            define_macros=[('USE_GPU', '1'), ('GLOG_USE_GLOG_EXPORT', '1')])
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
else:
    setup(name='torch_sht', ext_modules=[
        cpp_extension.CppExtension('torch_sht', ['sht.cpp'],
            libraries=['shtns_omp', 'fftw3', 'fftw3_threads'])
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
