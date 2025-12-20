# setup.py
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import os, sys, platform

def get_compile_args():
    """Return per-compiler optimized C/C++ flags."""
    if sys.platform.startswith("win"):
        # MSVC
        return ["/O2", "/DNDEBUG"]
    # GCC/Clang
    args = ["-O3", "-fno-math-errno", "-fno-trapping-math", "-DNDEBUG"]
    # Use -march=native only for local/non-portable builds
    if os.environ.get("PORTABLE_BUILD") != "1":
        # (Clang on macOS largely ignores -march=native; harmless)
        args.append("-march=native")
    return args

class BuildExt(build_ext):
    """Inject flags into every extension in one place."""
    def build_extensions(self):
        flags = get_compile_args()
        for ext in self.extensions:
            ext.extra_compile_args = (ext.extra_compile_args or []) + flags
        super().build_extensions()

ext_modules = [
    Extension(
        name="ngramprep.ngram_filter.filters.core_cy",
        sources=["src/ngramprep/ngram_filter/filters/core_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="daviesprep.davies_filter.filters.core_cy",
        sources=["src/daviesprep/davies_filter/filters/core_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
]

compiler_directives = dict(
    language_level=3,
    boundscheck=False,
    wraparound=False,
    cdivision=True,
    initializedcheck=False,
    nonecheck=False,
    embedsignature=False,
)

setup(
    name="ngram-kit",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=cythonize(ext_modules, compiler_directives=compiler_directives),
    cmdclass={"build_ext": BuildExt},
)
