#!/bin/sh

python3 ./main.py --model 'vgg16' --run_mode 'cam' --DATASET 'cifar10' --BATCH_SIZE 1 --target_classes 'aeroplane' --bg_classes 'cat'  --starting_parameters '/model_eoe_70.pth' --outf './cifar10/vgg16/output'

python3 ./main.py --model 'vgg16' --run_mode 'cam' --DATASET 'pascal' --BATCH_SIZE 1 --target_classes 'aeroplane' --bg_classes 'cat'  --starting_parameters '/model_eoe_65.pth' --outf './pascal/vgg16/output'
