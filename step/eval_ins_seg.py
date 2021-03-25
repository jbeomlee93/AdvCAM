
import numpy as np
import os

import chainercv
from chainercv.datasets import VOCInstanceSegmentationDataset

def run(args):
    dataset = VOCInstanceSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)

    pred_class = []
    pred_mask = []
    pred_score = []
    gt_masks = []
    gt_labels = []
    out_dir = None
    n_img = 0
    for i, id in enumerate(dataset.ids):
        print(id)
        if (out_dir is not None) and (not os.path.exists(os.path.join(out_dir, id + '.npy'))):
            continue
        gt_masks.append(dataset.get_example_by_keys(i, (1,))[0])
        gt_labels.append(dataset.get_example_by_keys(i, (2,))[0])
        ins_out = np.load(os.path.join(args.ins_seg_out_dir, id + '.npy'), allow_pickle=True).item()
        pred_class.append(ins_out['class'])
        pred_mask.append(ins_out['mask'])
        pred_score.append(ins_out['score'])
        n_img += 1
    print(n_img)
    print('0.5iou:', chainercv.evaluations.eval_instance_segmentation_voc(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.5))
    print('0.75iou:', chainercv.evaluations.eval_instance_segmentation_voc(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.75))

