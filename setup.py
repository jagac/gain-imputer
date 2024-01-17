from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="gain_imputer",
    version="0.0.10",
    description="Missing tabular data imputation using GANs",
    package_dir={"": "gain_imputer"},
    packages=find_packages(where="gain_imputer"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jagac/gain-imputer",
    author="Jagac",
    author_email="jagac41@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["tqdm>=4.66.1", "numpy>=1.26.2", "torch>=2.1.2"],
    python_requires=">=3.8",
)