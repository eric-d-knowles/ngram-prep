"""Setup file for building Cython extensions."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# Define extensions
extensions = [
    Extension(
        "ngramprep.ngram_filter.filters.core_cy",
        sources=[
            "src/ngramprep/ngram_filter/filters/core_cy.pyx",
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
    Extension(
        "daviesprep.davies_filter.filters.core_cy",
        sources=[
            "src/daviesprep/davies_filter/filters/core_cy.pyx",
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
