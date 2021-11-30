from distutils.core import setup

import numpy
from Cython.Build import cythonize


setup(
    name="train_models",
    ext_modules=cythonize("src/modelling/train_models.py", 
                          include_path=[numpy.get_include()],
                          compiler_directives={"language_level" : "3"})
)