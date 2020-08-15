from setuptools import setup
#from .pySLM2.version import __version__

version = {}
with open("pySLM2/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='pySLM2',
    version=version['__version__'],
    packages=['pySLM2'],
    url='',
    license='',
    author='Gilbert',
    author_email='',
    description=''
)
