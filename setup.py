#!/usr/bin/env python

from setuptools import setup

version = '1.0.0'

required = open('requirements.txt').read().split('\n')

setup(
    name='pcshrink',
    version=version,
    description=' ',
    author='Joseph Marcus / Chi-Chun Liu',
    author_email='jhmarcus@uchicago.edu / chichun@uchicago.edu',
    url='https://github.com/jhmarcus/pcshrink',
    packages=['pcshrink'],
    install_requires=required,
    long_description='See ' + 'https://github.com/jhmarcus/pcshrink',
    license='MIT'
)
