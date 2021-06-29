from setuptools import setup, find_packages

DEV_REQUIRES = ["black", "shap"]

setup(
    name="shappack",
    version="0.1.0",
    # description='',
    author="Hirofumi Tsuruta",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=["numpy", "pandas", "sklearn", "scipy"],
    extras_require={
        "dev": DEV_REQUIRES,
    },
)
