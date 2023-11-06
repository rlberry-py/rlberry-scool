from setuptools import setup, find_packages
import os

ver_file = os.path.join("rlberry_scool", "_version.py")
with open(ver_file) as f:
    exec(f.read())

packages = find_packages(exclude=["docs"])

#
# Base installation (interface only)
#
install_requires = ["rlberry"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rlberry-scool",
    version=__version__,
    description="Teaching Reinforcement Learning made easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Inria Scool team",
    url="https://github.com/rlberry-py",
    license="MIT",
    packages=packages,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    zip_safe=False,
)
