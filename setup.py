import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

libs = ['GLEW', 'GL', 'CGAL','CGAL_Core','gmp']

USE_ENERGY_STUFF = True
if USE_ENERGY_STUFF:
    assert os.path.exists('thirdparty/MRF2.2/libMRF.a'), 'Must build MRF lib first!'
    libs += ['MRF']

setup(name='pymeshedup_c',
        ext_modules=[
            cpp_extension.CppExtension(
                'pymeshedup_c',
                ['src/binding.cc', 'src/octree.cc',
                 'src/mesh.cc', 'src/dt.cc',
                 'src/mfmc.cc', 'src/vu.cc',
                 'src/twod_mrf.cc'],
                include_dirs=['/usr/local/include/eigen3', os.getcwd()+'/thirdparty/MRF2.2/'],
                extra_compile_args=['-O3', '-fopenmp'],
                library_dirs=['/usr/lib/x86_64-linux-gnu/', './thirdparty/MRF2.2/'],
                libraries=libs,
                )],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
