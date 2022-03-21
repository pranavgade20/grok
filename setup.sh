#!/bin/bash

/usr/bin/python3.9 -m pip install --upgrade pip
ln -sf /usr/bin/python3.9 /usr/bin/python
python setup.py install
pip install -e .
