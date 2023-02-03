import os.path

import numpy as np
import torch.optim
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.config import CfgNode

from detectron2.checkpoint import DetectionCheckpointer

from modeling.main_model import RetinaNetQueryDet
from modeling.main_model import add_querydet_config

import matplotlib.pyplot as plt
from dataloader import build_eval_loader

from detectron2.utils.events import EventStorage

import cv2
from pycocotools.coco import COCO

from fvcore.common.param_scheduler import MultiStepParamScheduler

import time
from torch.cuda.amp import autocast

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_querydet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def show_batch_gt_box(batch_data):
    for data in batch_data:
        img = np.ascontiguousarray(np.transpose(data[0]['image'].numpy(), [1, 2, 0]))
        anns = data[0]['instances']._fields['gt_boxes']
        for ann in anns:
            x1, y1, x2, y2 = ann.T.numpy()
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(img)
        plt.show()

def show_results(single_data, single_result, thres_fix = False, thres = 0.4, show_gt = False, save_dir = None):
    for data, result in zip(single_data, single_result):
        print(data['file_name'])
        img = np.ascontiguousarray(np.transpose(data['image'].numpy(), [1, 2, 0]))
        anns = result['instances']._fields['pred_boxes']
        scores = result['instances']._fields['scores']
        pred_classes = result['instances']._fields['pred_classes']
        if thres_fix is not False:
            top_idx = scores > thres
            anns = anns[top_idx]
            scores = scores[top_idx]
            pred_classes = pred_classes[top_idx]
            x_scale_factor = img.shape[0] / data['height']
            y_scale_factor = img.shape[1] / data['width']
        if show_gt:
            gt_anns = data['instances']._fields['gt_boxes']
            gt_classes = data['instances']._fields['gt_classes']
            for gt_ann, gt_class in zip(gt_anns, gt_classes):
                x1, y1, x2, y2 = gt_ann.T.numpy()
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                img = cv2.putText(img, 'gt_class:{}'.format(str(int(gt_class))), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        for ann, score, pred_class in zip(anns, scores, pred_classes):
            x1, y1, x2, y2 = ann.detach().T.numpy()
            if thres_fix is not False:
                x1 *= x_scale_factor
                x2 *= x_scale_factor
                y1 *= y_scale_factor
                y2 *= y_scale_factor
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            img = cv2.putText(img, 'class:{}'.format(str(int(pred_class))), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            img = cv2.putText(img, 'score:{:.4f}'.format(float(score)), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if save_dir is None:
            plt.rcParams['figure.figsize'] = (20.0, 20.0)
            plt.imshow(img)
            plt.show()
        if save_dir is not None:
            if not os.path.exists(save_dir): os.mkdir(save_dir)
            assert isinstance(save_dir, str)
            cv2.imwrite(os.path.join(save_dir, data['file_name'].split('\\')[-1]), img)

if __name__ == "__main__":
    cfg = get_cfg()  # cfg默认配置来自D:\Code\coding-py\detectron2\detectron2\config\defaults.py
    add_querydet_config(cfg)
    cfg.merge_from_file('./config/test.yaml')
    cfg.freeze()
    default_setup(cfg, None)

    # 构建数据集读取器
    dataloader, datalist_len = build_eval_loader(cfg)
    data_loader_obj = iter(dataloader)
    # batch_data = [next(data_loader_obj) for _ in range(5)]

    # coco = COCO(cfg.VISDRONE.TRAIN_JSON)
    # show_batch_gt_box(batch_data) # 抽样了一下，虽然标注有点差，但是输入应该没什么问题。

    # 构建模型
    model = RetinaNetQueryDet(cfg)
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    # checkpointer.resume_or_load(cfg.MODEL.WEIGHTS)
    checkpointer.resume_or_load(cfg.OUTPUT_DIR + '/119000.pth')
    # 构建优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)

    # 模型参数量测试
    single_data = next(data_loader_obj) # image.shape = (3, 1125, 1999)
    with EventStorage(start_iter=0) as storage:
        for i in range(datalist_len):
            result = model.test(single_data)
            show_results(single_data, result, thres_fix=True, show_gt=True, save_dir='output_single')
            single_data = next(data_loader_obj)


    # print(result)
    # show_batch_gt_box(single_data)

    # 批量测试
