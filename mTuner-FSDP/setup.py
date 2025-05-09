import os
import subprocess
import sys

from setuptools import setup, find_packages


setup(
    name="FineTuneHub",
    version="0.0.1",
    packages=find_packages(include=['fthub', 'fthub.*']),
)
