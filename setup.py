from setuptools import setup, find_packages

DEV_REQUIRES = ["black"]

setup(
    name='shappack',
    version='0.1.0',
    #description='',
    author='Hirofumi Tsuruta',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
    ],
    extras_require={
        "dev": DEV_REQUIRES,
    },
)
