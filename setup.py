from setuptools import setup, find_packages

setup(
    name='shappack',
    version='0.1.0',
    #description='',
    author='Hirofumi Tsuruta',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
    ]
)
