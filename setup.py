"""
Setup script for SAFE fraud detection package
"""

from setuptools import setup, find_packages
import os


def read_requirements():
    """Read requirements from requirements.txt"""
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def read_long_description():
    """Read long description from README"""
    if os.path.exists('README.md'):
        with open('README.md', encoding='utf-8') as f:
            return f.read()
    return ""


setup(
    name='safe-fraud-detection',
    version='1.0.0',
    author='rotiyan',
    description='A Neural Survival Analysis Model for Fraud Early Detection',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/rotiyan/hazard_rate',
    license='MIT',
    # Exclude safe_fraud_detection.tests too, not just a top-level tests package
    packages=find_packages(exclude=['tests', 'tests.*', '*.tests', '*.tests.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
            'build>=1.0.0',
        ],
        'viz': [
            'matplotlib>=3.7.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'safe-train=safe_fraud_detection.scripts.train:main',
            'safe-eval=safe_fraud_detection.scripts.evaluate:main',
        ],
    },
    include_package_data=True,
    # MANIFEST.in keeps tests in the sdist; don't install them as package data
    exclude_package_data={'safe_fraud_detection': ['tests/*', 'tests/*/*']},
    zip_safe=False,
)
