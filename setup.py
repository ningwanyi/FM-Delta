import os
import setuptools
import sys
import numpy as np

# cython -3 --fast-fail -v --cplus ./fmd.pyx

join = os.path.join
fmddir = 'package'

sources = [
    join(fmddir, 'src', x) for x in (
        'error.cpp', 'rcdecoder.cpp', 'rcencoder.cpp',
        'rcqsmodel.cpp', 'write.cpp', 'read.cpp',
    )
]
sources += ['fmd.pyx']

extra_compile_args = ['-O3', '-std=c++11']

setuptools.setup(
    name="fmd",
    setup_requires=['numpy'],
    python_requires="~=3.6",  # >= 3.6 < 4.0
    ext_modules=[
        # cythonize('fmd.pyx'),
        setuptools.Extension(
            'fmd',
            sources=sources,
            language='c++',
            include_dirs=[join(fmddir, 'include'), np.get_include()],
            extra_compile_args=extra_compile_args,
            library_dirs=[], 
            libraries=[],  
            extra_link_args=[],
        )
    ],
)
