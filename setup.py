from setuptools import find_packages, setup


setup(
    name="PeakGemm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["ninja", "torch"],
)
