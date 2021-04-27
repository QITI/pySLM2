from setuptools import setup

version = {}
with open("pySLM2/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='pySLM2',
    version=version['__version__'],
    packages=['pySLM2','pySLM2.util'],
    url='https://github.com/QITI/pySLM2',
    license='',
    author='Chung-You (Gilbert) Shih',
    author_email='c5shih@uwaterloo.ca',
    description=''
)
