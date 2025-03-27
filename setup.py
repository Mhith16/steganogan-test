#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

install_requires = [
    'imageio>=2.19.0',
    'reedsolo>=1.5.3',
    'scipy>=1.8.0',
    'tqdm>=4.64.0',
    'numpy>=1.22.0',
    'Pillow>=9.0.0',
    'torch>=1.12.0',
    'torchvision>=0.13.0',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=7.0.0',
    'pytest-cov>=2.12.0',
]

development_requires = [
    # general
    'pip>=21.0',
    'bumpversion>=0.5.3',
    'watchdog>=0.8.3',

    # docs
    'Sphinx>=4.0.0',
    'sphinx_rtd_theme>=1.0.0',

    # style check
    'flake8>=4.0.0',
    'isort>=5.0.0',

    # fix style issues
    'autoflake>=1.4',
    'autopep8>=1.6.0',

    # distribute on PyPI
    'twine>=4.0.0',
    'wheel>=0.37.0',

    # Advanced testing
    'coverage>=6.0',
    'tox>=3.24.0',

    # Notebooks
    'jupyter>=1.0.0',
]

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Steganography tool for medical images based on DeepLearning GANs",
    entry_points={
        'console_scripts': [
            'steganogan=steganogan.cli.main:main'
        ],
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='steganogan',
    name='steganogan',
    packages=find_packages(include=['steganogan', 'steganogan.*']),
    python_requires='>=3.7',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/DAI-Lab/SteganoGAN',
    version='0.2.0',
    zip_safe=False,
)