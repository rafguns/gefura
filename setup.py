from setuptools import setup

long_description = open('README.md', encoding='utf-8').read()

setup(
    name='gefura',
    version='0.1',
    url='http://github.com/rafguns/gefura/',
    license='New BSD License',
    author='Raf Guns',
    tests_require=['pytest', 'pytest-cov'],
    install_requires=[
        'networkx>=2.8',
    ],
    author_email='raf.guns@uantwerpen.be',
    description='Discover \'bridges\' between node groups in a network',
    long_description=long_description,
    py_modules=['gefura'],
    platforms='any',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ]
)
