#!/bin/bash

PATH=$PWD/../data/train/
FUNC_LIST_FILE=$PWD/../data/train/function-word-list.txt
FILE_PATH=$PWD/../data/train/contexts/

python induction.py --directory $PATH --funcWordFile $FUNC_LIST_FILE --filePath $FILE_PATH