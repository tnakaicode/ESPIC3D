#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

py.test --genscript=mypytestscript

python3 mypytestscript $DIR/testEsSolve.py $DIR/testLinAlgSolve.py
