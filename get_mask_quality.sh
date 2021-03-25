#!/bin/bash

python obtain_CAM_masking.py --train_list voc12/train_aug.txt
python run_sample.py --eval_cam_pass True --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True
