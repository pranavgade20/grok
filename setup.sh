#!/bin/bash

/usr/bin/python3.9 -m pip install --upgrade pip
pip install setuptools==59.5.0
ln -sf /usr/bin/python3.9 /usr/bin/python
python setup.py install
pip install -e .
