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
import pycocotools.coco, pycocotools.cocoeval

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


def show_results(single_data, single_result, thres_fix=False, thres=0.4, show_gt=False):
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
                img = cv2.putText(img, 'gt_class:{}'.format(str(int(gt_class))), (int(x1), int(y1)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        for ann, score, pred_class in zip(anns, scores, pred_classes):
            x1, y1, x2, y2 = ann.detach().T.numpy()
            if thres_fix is not False:
                x1 *= x_scale_factor
                x2 *= x_scale_factor
                y1 *= y_scale_factor
                y2 *= y_scale_factor
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            img = cv2.putText(img, 'class:{}'.format(str(int(pred_class))), (int(x1), int(y1)),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            img = cv2.putText(img, 'score:{}'.format(str(float(score))), (int(x1), int(y1) + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(img)
        plt.show()


def build_coco_results(pred_list, coco_gt, thres_fix=False, thres=0.4):
    coco_pred = pycocotools.coco.COCO()
    coco_pred.dataset = {}
    coco_pred.dataset['images'] = []
    coco_pred.dataset['annotations'] = []
    coco_pred.dataset['categories'] = coco_gt.dataset['categories']

    coco_gt_fix = pycocotools.coco.COCO()
    coco_gt_fix.dataset = {}
    coco_gt_fix.dataset['images'] = []
    coco_gt_fix.dataset['annotations'] = []

    pred_anno_id = 0
    gt_anno_id = 0
    img_id = 0
    for pred_list_idx, (batch_data, batch_result) in enumerate(pred_list):
        assert len(batch_result) == len(batch_data)
        for result_idx in range(len(batch_result)):
            result = batch_result[result_idx]
            data = batch_data[result_idx]
            x_scale_factor = result['instances'].image_size[0] / data['height']
            y_scale_factor = result['instances'].image_size[1] / data['width']
            anns = result['instances']._fields['pred_boxes']
            scores = result['instances']._fields['scores']
            pred_classes = result['instances']._fields['pred_classes']
            # 从dataloader中构建coco_gt。不能直接用files构建coco_gt，因为images_id不一致。
            for gt_box, gt_class in zip(data['instances']._fields['gt_boxes'], data['instances']._fields['gt_classes']):
                x1, y1, x2, y2 = gt_box.T.numpy()
                coco_gt_fix.dataset['annotations'].append({
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0,
                    'image_id': data['image_id'],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'category_id': int(gt_class),
                    'id': gt_anno_id
                })
                gt_anno_id += 1
            coco_gt_fix.dataset['images'].append({
                'file_name': data['file_name'],
                'height': data['height'],
                'width': data['width'],
                'image_id': data['image_id'],
                'id': img_id
            })

            cpu_device = torch.device('cpu')
            for ann, score, pred_class in zip(anns, scores, pred_classes):
                if thres_fix and score < thres:
                    continue
                x1, y1, x2, y2 = torch.Tensor.cpu(ann.detach().T).numpy()
                x1 = max(0, x1) * x_scale_factor
                y1 = max(0, y1) * y_scale_factor
                x2 = min(data['width'], x2) * x_scale_factor
                y2 = min(data['height'], y2) * y_scale_factor
                coco_pred.dataset['annotations'].append({
                    'area': (x2 - x1) * (y2 - y1),
                    'id': pred_anno_id,  # annotation的id是全局的，每个都不一样。通过image_id区分不同图片的annotation
                    'image_id': data['image_id'],
                    'category_id': int(pred_class),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(score)
                })
                pred_anno_id += 1
            coco_pred.dataset['images'].append({
                'id': img_id,
                'image_id': data['image_id'],
                'width': data['width'],
                'height': data['height'],
                'file_name': data['file_name']
            })
            img_id += 1

    coco_pred.dataset['categories'] = coco_gt.dataset['categories']
    coco_pred.createIndex()
    coco_gt_fix.dataset['categories'] = coco_gt.dataset['categories']
    coco_gt_fix.createIndex()
    return coco_pred, coco_gt_fix


import pickle
from make_result_json import make_result_json

def eval_func(model_name):
    cfg = get_cfg()  # cfg默认配置来自D:\Code\coding-py\detectron2\detectron2\config\defaults.py
    add_querydet_config(cfg)
    cfg.merge_from_file('./config/test.yaml')
    cfg.freeze()
    default_setup(cfg, None)

    dataloader, dataloader_len = build_eval_loader(cfg)  # 这个loader包装的是SequentialSampler，所以是保证全部顺序读取的
    dataloader_obj = iter(dataloader)

    model = RetinaNetQueryDet(cfg)
    cuda_device = torch.device('cuda:1')
    model.to(cuda_device)
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    # checkpointer.resume_or_load(cfg.MODEL.WEIGHTS)
    checkpointer.resume_or_load(cfg.OUTPUT_DIR + model_name)

    model.eval()
    pred_list = []
    # with EventStorage(start_iter=0) as storage:
    with torch.no_grad():
        data_obj = next(dataloader_obj)
        for i in range(dataloader_len):
            result = model.test(data_obj)
            print("evaluating {} ...".format(str(i)))
            pred_list.append([data_obj, result])
            try:
                data_obj = next(dataloader_obj)
            except StopIteration:
                print('All sample evaluated. StopIteration.')
                break
    return pred_list
    # make_result_json(pred_list, thres_fix=True, thres=0.3, model_name=model_name)

import os
import torch.multiprocessing

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    output_dir = './running_output'
    model_list = [model_name for model_name in os.listdir(output_dir) if model_name.endswith('.pth')]
    for model_name in model_list:
        # model_idx = i * 10000
        # model_name = '{}.pth'.format(str(model_idx))
        pred_list = eval_func(model_name)
        make_result_json(pred_list.copy(), thres_fix=False, model_name=model_name)
        # for i in range(1, 20):
            # thres = i * 0.05
            # make_result_json(pred_list.copy(), thres_fix=True, thres=thres, model_name=model_name)