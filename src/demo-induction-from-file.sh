#!/bin/bash

WORK_PATH=$PWD/../data/train/
FUNC_LIST_FILE=$PWD/../data/train/function-word-list.txt
FILE_WORK_PATH=$PWD/../data/train/contexts/

python induction-from-file.py --directory $WORK_PATH --funcWordFile $FUNC_LIST_FILE --filePath $FILE_WORK_PATH