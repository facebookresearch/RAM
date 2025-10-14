"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(requirements_path: Path):
    if not requirements_path.exists():
        return []
    lines = []
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


setup(
    name="otbench",
    version="0.1.0",
    packages=find_packages(
        include=["otbench", "otbench.*", "otb_creation", "otb_creation.*"]
    ),
    include_package_data=True,
    install_requires=read_requirements(Path(__file__).parent / "requirements.txt"),
    entry_points={
        "console_scripts": [
            "otbench=otbench.cli:main",
        ]
    },
)
