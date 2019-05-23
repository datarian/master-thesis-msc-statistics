from setuptools import setup, find_packages
from distutils.core import setup
import distutils.command.bdist_conda

__version__ = '0.4'

# Get the long description from the README file
with open('README.md') as f:
    long_description = f.read()

setup(
    name='kdd98',
    version=__version__,
    description='Data handler for the KDD98 data sets',
    long_description=long_description,
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    author='Florian Hochstrasser',
    install_requires=[
      'numpy>=1.11.1',
      'scikit-learn>=0.20.2',
      'scipy>=0.17.0',
      'statsmodels>=0.6.1',
      'pandas>=0.21.1',
      'patsy>=0.4.1',
      'category_encoders>=2.0.0',
      'geopy',
      'python-dateutil>=2.8.0',
    ],
    author_email='datarian@againstthecurrent.ch'
)
