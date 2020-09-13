from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pymeshedup_c',
        ext_modules=[
            cpp_extension.CppExtension(
                'pymeshedup_c',
                ['src/binding.cc', 'src/octree.cc',
                 'src/mesh.cc', 'src/dt.cc',
                 'src/mfmc.cc', 'src/vu.cc'],
                include_dirs=['/usr/local/include/eigen3'],
                extra_compile_args=['-O3', '-fopenmp'],
                library_dirs=['/usr/lib/x86_64-linux-gnu/'],
                libraries=['GLEW', 'GL', 'CGAL','CGAL_Core','gmp'],
                )],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
