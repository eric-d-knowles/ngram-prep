"""Setup file for building Cython extensions."""

import shutil
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize


def check_compiler():
    if shutil.which("g++") is None:
        sys.exit(
            "\nError: g++ not found.\n"
            "Building lexichron requires a C++ compiler. "
            "If you are using conda, install one with:\n\n"
            "    conda install -c conda-forge gxx\n\n"
            "Then retry: pip install .\n"
        )


check_compiler()

extensions = [
    Extension(
        "ngramprep.ngram_filter.filters.core_cy",
        sources=["src/ngramprep/ngram_filter/filters/core_cy.pyx"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
    Extension(
        "daviesprep.davies_filter.filters.core_cy",
        sources=["src/daviesprep/davies_filter/filters/core_cy.pyx"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
