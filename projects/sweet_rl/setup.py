"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC-By-NC license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="sweet_rl",
    packages=["sweet_rl"],
    install_requires=open("requirements.txt", "r").read().split(),
    include_package_data=True,
)
