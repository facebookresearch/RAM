from setuptools import setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='sweet_rl',
    packages=['sweet_rl'],
    install_requires=open("requirements.txt", "r").read().split(),
    include_package_data=True,
)