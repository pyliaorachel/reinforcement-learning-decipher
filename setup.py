from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rl-decipher',
    version='0.1.0',
    description='Reinforcement learning for decipherment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pyliaorachel/reinforcement-learning-decipher',
    author='Peiyu Liao',
    keywords='reinforcement-learning',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['gym', 'numpy', 'torch', 'torchvision'],
)
