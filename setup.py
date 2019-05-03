from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='line_convolution_cuda',
    ext_modules=[
        CUDAExtension('line_convolution_cuda', [
            'onehot.cpp',
            'onehot_kernel.cu',
            'gather.cpp',
            'gather_kernel.cu',
            'scatter.cpp',
            'scatter_kernel.cu',
            'cublas_wrapper.cpp'
        ],
        extra_compile_args=['-std=c++11']),
    ],
    cmdclass={'build_ext': BuildExtension}
)

