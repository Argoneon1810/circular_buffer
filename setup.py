from setuptools import setup, find_packages
import os

setup(
    name="circular-buffer",
    version="0.1.0",
    description="A flexible Circular Buffer implementation supporting both PyTorch and NumPy backends automatically.",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Argoneon1810",
    author_email="jinhoon.choi.97@gmail.com",
    url="https://github.com/Argoneon1810/circular-buffer",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)