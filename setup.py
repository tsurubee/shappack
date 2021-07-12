from setuptools import setup, find_packages

TEST_REQUIRES = ["black", "pytest"]
DEV_REQUIRES = TEST_REQUIRES + ["shap", "matplotlib", "jupyterlab"]


setup(
    name="shappack",
    version="0.1.0",
    description="Interpretable machine learning based on Shapley values",
    author="Hirofumi Tsuruta",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=["numpy", "pandas", "sklearn", "scipy"],
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
    },
)
