from __future__ import annotations

import os
import warnings

try:
    from setuptools._distutils.errors import (
        CCompilerError,
        CompileError,
        DistutilsExecError,
        DistutilsPlatformError,
        LinkError,
    )
except ImportError:  # pragma: no cover - older setuptools fallback
    from distutils.errors import (
        CCompilerError,
        CompileError,
        DistutilsExecError,
        DistutilsPlatformError,
        LinkError,
    )

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build the optional native accelerator when a compiler is available."""

    def run(self):
        try:
            super().run()
        except (
            DistutilsPlatformError,
            DistutilsExecError,
            CCompilerError,
            CompileError,
            LinkError,
            OSError,
        ) as exc:
            warnings.warn(f"Skipping native extension build: {exc}", RuntimeWarning)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except (
            DistutilsPlatformError,
            DistutilsExecError,
            CCompilerError,
            CompileError,
            LinkError,
            OSError,
        ) as exc:
            warnings.warn(f"Skipping native extension {ext.name}: {exc}", RuntimeWarning)


ext_modules = []
if os.environ.get("TERKET_DISABLE_NATIVE") != "1":
    include_dirs = [
        path
        for path in os.environ.get("TERKET_PYTHON_INCLUDE", "").split(os.pathsep)
        if path
    ]
    ext_modules.append(
        Extension(
            "terket._schur_native",
            sources=[
                "src/terket/_schur_native.c",
                "src/terket/_schur_native_support.c",
                "src/terket/_schur_native_algebra.c",
                "src/terket/_schur_native_graph.c",
                "src/terket/_schur_native_dp.c",
            ],
            include_dirs=include_dirs,
        )
    )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
)
