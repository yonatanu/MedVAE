from setuptools import setup

_REQUIRED = [
    "pandas>=1.3.5",
    "numpy>=1.18.0",
    ""
]

setup(
    name="medvae",
    version="0.0.1",
    description="MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders",
    author="Maya Varma, Ashwin Kumar, Rogier van der Sluijs",
    url="https://github.com/StanfordMIMI/medvae",
    packages=["medvae"],
    install_requires=_REQUIRED,
)