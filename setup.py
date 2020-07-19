import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roc2pr",
    version="0.0.1",
    author="Ameya Daigavane",
    author_email="ameya.d.98@gmail.com",
    description="A package to interconvert between ROC curves and PR curves.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ameya98/roc2pr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)