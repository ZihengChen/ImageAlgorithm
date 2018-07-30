import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ImageAlgoKD",
    version="0.0.2",
    author="Ziheng Chen",
    author_email="zihengchen2015@u.northwestern.edu",
    description="implementation of clustering by fast search and find density peak. Implementation is based on numpy, cuda and opencl",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZihengChen/ImageAlgorithm",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)