from setuptools import setup

long_description = open('README.md', encoding='utf-8').read()

setup(
    name='gefura',
    version='0.1',
    url='http://github.com/rafguns/gefura/',
    license='New BSD License',
    author='Raf Guns',
    tests_require=['nose'],
    install_requires=[
        'networkx>=2.5',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ]
)
