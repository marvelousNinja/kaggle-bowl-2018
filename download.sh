#!/bin/bash
set -e
set -v

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
cd ./data

kaggle competitions download -w -c data-science-bowl-2018

unzip stage1_test.zip -d ./test
unzip stage1_train.zip -d ./train
unzip stage1_train_labels.csv.zip -d ./train
