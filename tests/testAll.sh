#!/bin/bash

py.test --genscript=mypytestscript

python3 mypytestscript tests/testEsSolve.py tests/testLinAlgSolve.py
