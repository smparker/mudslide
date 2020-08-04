from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('mudslide/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

def readme():
    with open("README.md") as f:
        return f.read()

setup(name='mudslide',
    version=main_ns['__version__'],
    description='Quantum-Classical Mudslides',
    packages=['mudslide'],
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
    zip_safe=False
)

