from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('mudslide/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name='mudslide',
    packages=['mudslide'],
    version=main_ns['__version__'],
    license='MIT',
    description='Package to simulate nonadiabatic molecular dynamics using trajectory methods',
    author='Shane M. Parker',
    author_email='shane.parker@case.edu',
    url='https://github.com/smparker/mudslide',
    download_url='https://github.com/smparker/mudslide/archive/v0.9.tar.gz',
    keywords= ['science', 'chemistry', 'nonadiabatic dynamics'],
    install_requires=[
        'numpy>=1.19',
        'scipy',
        'typing_extensions'
        ],
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points={
        'console_scripts': [
            'mudslide = mudslide.__main__:main',
            'mudslide-surface = mudslide.surface:main'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
        ]
)

