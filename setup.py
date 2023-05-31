import setuptools
from time_series_models import version

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requires = fh.read().split("\n")


setuptools.setup(
    name="time_series_models",
    version=version.__version__,
    author="Camus Energy",
    author_email="seto-2243@camus.energy",
    description="Time series forecasting framework build around sci-kit-learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.camus.energy/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=requires,
    include_package_data=True,
    extras_require={
        "dev": [
            "black",
        ]
    },
    python_requires="~=3.10.6",
)
