from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.2"
DESCRIPTION = "Build brain circuits"
LONG_DESCRIPTION = "A package that allows to build simple brain circuits."

# Setting up
setup(
    name="compbrain",
    version=VERSION,
    author="Bo Yuan",
    author_email="by2291@columbia.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "tqdm~=4.64.0",
        "matplotlib~=3.5.1",
        "setuptools~=61.2.0",
        "pyyaml~=6.0"
    ],
    keywords=["python", "neurons", "synapses", "circuit"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        'Topic :: Scientific/Engineering :: Neuroscience',
    ]
)