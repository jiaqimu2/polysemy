#!/bin/bash

PATH=$PWD/../data/train/
FUNC_LIST_FILE=$PWD/../data/train/function-word-list.txt
POLY_LIST_FILE=$PWD/../data/train/poly-list.txt

python representation.py --directory $PATH --funcWordFile $FUNC_LIST_FILE --polyListFile $POLY_LIST_FILE