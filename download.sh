#!/bin/bash
set -e
set -v
set -a
. ./.env
set +a

cd $DATA_DIR

kg download -u $KAGGLE_USERNAME -p $KAGGLE_PASSWORD -c $KAGGLE_COMPETITION
unzip stage1_test.zip -d ./test
unzip stage1_train.zip -d ./train
unzip stage1_train_labels.csv.zip -d ./train
