import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="luparnet",
    version="0.0.1",
    author="Nick Lupariello",
    author_email="nicklupe13@gmail.com",
    description="A small basic neural network package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lupeboy2/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
