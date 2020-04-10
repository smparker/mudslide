from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('fssh/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

def readme():
    with open("README.md") as f:
        return f.read()

setup(name='fssh',
    version=main_ns['__version__'],
    description='Quantum-Classical Mudslides',
    packages=['fssh'],
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points={
        'console_scripts': [
            'fssh = fssh.__main__:main',
            'surface = fssh.surface:main'
        ]
    },
    zip_safe=False
)

