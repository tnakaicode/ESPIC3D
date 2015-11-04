#!/bin/bash

py.test --genscript=mypytestscript

python3 mypytestscript test/testEsSolve.py
