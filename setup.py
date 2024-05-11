import pathlib

from setuptools import find_packages, setup

__version__ = "0.0.0+dev0"


with open("requirements.txt") as f:
    install_requires = [x.strip() for x in f.readlines()]

LICENSE: str = "Apache"
README: str = pathlib.Path("README.md").read_text()

setup(
    name="critical-dream",
    version=__version__,
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    long_description=README,
    long_description_content_type="text/markdown",
    license=LICENSE,
    packages=find_packages(
        include=["critical_dream*"],
    ),
    python_requires=">3.7",
    platforms="any",
    install_requires=install_requires,
)
