"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="ram",
    version="0.1",
    package_dir={"ram": "ram"},
    packages=find_packages(include=["ram.*"]),
    description="Open source code for projects by the RAM team",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Fundamental AI Research (FAIR) at Meta",
    url="https://github.com/facebookresearch/RAM",
    license="MIT",
    install_requires=[
        "alpaca_eval",
        "datasets",
        "pre-commit",
        "black==24.8.0",
        "isort==5.13.2",
    ],
)
