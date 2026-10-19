from setuptools import setup, find_packages

setup(
    name="love",
    version="1.0.0",
    description="LOVE: Latent-model based OVErlapping clustering",
    author="Xin Bing",
    author_email="xb43@cornell.edu",
    url="https://github.com/bingx1990/LOVE",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "networkx>=2.6",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
