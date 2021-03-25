#!/bin/bash

python obtain_CAM_masking.py --train_list voc12/train.txt
python run_sample.py --eval_cam_pass True
