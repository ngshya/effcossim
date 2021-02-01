from setuptools import setup, find_packages
import os
import sys


if sys.version_info[0] < 3:
    with open('README.md') as f:
        long_description = f.read()
else:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

with(open("version.txt", "r")) as f:
    version = f.read()

with(open("requirements.txt", "r")) as f:
    requirements = f.read()


setup(
    name='effcossim',
    version=version,
    description='Efficient Pairwise Cosine Similarity Computation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='ngshya',
    author_email='ngshya@gmail.com',
    url='https://github.com/ngshya/sdaab',
    license='GPLv3',
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    include_package_data=True,
)