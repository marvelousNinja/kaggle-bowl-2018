#!/bin/bash
set -e
set -v
set -a
. ./.env
set +a

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle

cd $DATA_DIR

kaggle competitions download -w -c $KAGGLE_COMPETITION

unzip stage1_test.zip -d ./test
unzip stage1_train.zip -d ./train
unzip stage1_train_labels.csv.zip -d ./train
