import os
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

#libs = ['GLEW', 'GL', 'CGAL','CGAL_Core','gmp']
libs = ['GLEW', 'GL', 'gmp']

USE_ENERGY_STUFF = True
if USE_ENERGY_STUFF:
    assert os.path.exists('thirdparty/MRF2.2/libMRF.a'), 'Must build MRF lib first!'
    libs += ['MRF']

setup(name='pymeshedup_c',
        ext_modules=[
            #CUDAExtension(
            CppExtension(
                'pymeshedup_c',
                ['src/binding.cc', 'src/octree.cc',
                 'src/mesh.cc', 'src/dt.cc',
                 'src/mfmc.cc', 'src/vu.cc',
                 'src/twod_mrf.cc',
                 'src/tensor_octree.cc'],
                include_dirs=[
                    '/usr/local/include/eigen3',
                    os.getcwd()+'/thirdparty/MRF2.2/',
                    '/opt/CGAL-5.1/include'
                    ],
                extra_compile_args=['-O3', '-fopenmp'],
                library_dirs=['/usr/lib/x86_64-linux-gnu/', './thirdparty/MRF2.2/'],
                libraries=libs,
                )],
        cmdclass={'build_ext': BuildExtension})
