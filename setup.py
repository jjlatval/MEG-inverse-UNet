# NOTE: This setup.py is not yet working - do not use it!

from distutils.core import setup
from pip.req import parse_requirements
from pip.download import PipSession

requirements = [str(r.req) for r in parse_requirements('requirements.txt', session=PipSession)]

setup(name='meg_inverse',
      version='0.1',
      description='MEG inverse problem deep neural network solver',
      author='Joni Latvala',
      author_email='joni.latvala@gmail.com',
      packages=['meg_inverse'],
      install_requires=requirements,
      test_suite='meg_inverse.tests',
      use_2to3=True
      )
