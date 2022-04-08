#!/bin/sh

python3 ./main.py --model 'vgg16' --run_mode 'train' --DATASET 'dsiac' --BATCH_SIZE 1  --starting_parameters '0' --outf './dsiac/vgg16/output' --NUM_CLASSES 2 --EPOCHS 500 --LR 0.1
