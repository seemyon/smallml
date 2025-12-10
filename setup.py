"""Setup script for SmallML package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description_path = this_directory / "README.md"

# Only read README if it exists (for now it doesn't)
if long_description_path.exists():
    long_description = long_description_path.read_text(encoding='utf-8')
else:
    long_description = "SmallML: Bayesian Transfer Learning for Small-Data Predictive Analytics"

# Read version
version = {}
with open("smallml/version.py") as f:
    exec(f.read(), version)

setup(
    name="smallml",
    version=version['__version__'],
    author=version['__author__'],
    author_email=version['__email__'],
    description=version['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=version['__url__'],
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pymc>=5.0.0",
        "arviz>=0.22.0",
        "numpy>=2.3.0",
        "pandas>=2.3.0",
        "scikit-learn>=1.7.0",
        "scipy>=1.16.0",
    ],
    include_package_data=True,
    package_data={
        "smallml": ["data/*.pkl"],
    },
    keywords=[
        "machine-learning",
        "bayesian",
        "transfer-learning",
        "small-data",
        "hierarchical-models",
        "conformal-prediction",
    ],
    project_urls={
        "Bug Reports": "https://github.com/seemyon/smallml/issues",
        "Source": "https://github.com/seemyon/smallml",
    },
)
