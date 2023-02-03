import json
import torch
import os
import logging
import numpy as np
import copy

from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.structures.boxes import BoxMode
from detectron2.data import samplers
from detectron2.utils.env import seed_all_rng


class Mapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True, simple_transform=False):
        if simple_transform:
            self.tfm_gens = [T.ResizeShortestEdge(short_edge_length=cfg.VISDRONE.SHORT_LENGTH, max_size=cfg.VISDRONE.MAX_LENGTH,
                                 sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]
        else:
            self.tfm_gens = build_transform_gen(cfg, is_train)
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = False
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = False
        self.load_proposals = False
        self.keypoint_hflip_indices = None
        # fmt: on

        self.is_train = is_train

    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict)
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.img_format)
        detection_utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if False:# not self.is_train: # 这里有个is_train = False会MemeoryError问题
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                detection_utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = detection_utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
        return dataset_dict


def build_transform_gen(cfg, is_train):
    if is_train:
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        sample_style = 'choice'

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        tfm_gens.append(
            T.ResizeShortestEdge(short_edge_length=cfg.VISDRONE.SHORT_LENGTH, max_size=cfg.VISDRONE.MAX_LENGTH,
                                 sample_style=sample_style))
    else:
        tfm_gens.append(
            T.ResizeShortestEdge(short_edge_length=[cfg.VISDRONE.TEST_LENGTH], max_size=cfg.VISDRONE.TEST_LENGTH,
                                 sample_style=sample_style))

    return tfm_gens

def get_train_data_dicts(json_file, img_root, filter_empty=False):
    data = json.load(open(json_file))

    images = {x['id']: {'file': x['file_name'], 'height': x['height'], 'width': x['width']} for x in data['images']}

    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations.keys():
            annotations[img_id] = []
        annotations[img_id].append(
            {'bbox': ann['bbox'], 'category_id': ann['category_id'], 'iscrowd': ann['iscrowd'], 'area': ann['area']})

    for img_id in images.keys():
        if img_id not in annotations.keys():
            annotations[img_id] = []

    data_dicts = []
    for img_id in images.keys():
        if filter_empty and len(annotations[img_id]) == 0:
            continue
        data_dict = {}
        data_dict['file_name'] = str(os.path.join(img_root, images[img_id]['file']))
        data_dict['height'] = images[img_id]['height']
        data_dict['width'] = images[img_id]['width']
        data_dict['image_id'] = img_id
        data_dict['annotations'] = []
        for ann in annotations[img_id]:
            data_dict['annotations'].append(
                {'bbox': ann['bbox'], 'iscrowd': ann['iscrowd'], 'category_id': ann['category_id'] - 1,
                 'bbox_mode': BoxMode.XYWH_ABS})
        data_dicts.append(data_dict)
    return data_dicts


def get_test_data_dicts(json_file, img_root):
    data = json.load(open(json_file))
    images = {x['id']: {'file': x['file_name'], 'height': x['height'], 'width': x['width']} for x in data['images']}

    data_dicts = []
    for img_id in images.keys():
        data_dict = {}
        data_dict['file_name'] = str(os.path.join(img_root, images[img_id]['file']))
        data_dict['height'] = images[img_id]['height']
        data_dict['width'] = images[img_id]['width']
        data_dict['image_id'] = img_id
        data_dict['annotations'] = []
        data_dicts.append(data_dict)
    return data_dicts


def build_eval_loader(cfg):
    dataset_dicts = get_train_data_dicts(cfg.VISDRONE.TEST_JSON, cfg.VISDRONE.TEST_IMG_ROOT)
    dataset = DatasetFromList(dataset_dicts)
    mapper = Mapper(cfg, True, simple_transform=True)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.TrainingSampler(len(dataset), shuffle=False)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader, len(dataset)


def build_train_loader(cfg):
    images_per_worker = cfg.SOLVER.IMS_PER_BATCH # 每个batch的图片数

    dataset_dicts = get_train_data_dicts(cfg.VISDRONE.TRAIN_JSON, cfg.VISDRONE.TRING_IMG_ROOT) # 读取一个list
    dataset = DatasetFromList(dataset_dicts, copy=False)
    mapper = Mapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_worker, drop_last=True
    )
    # drop_last so the batch always have the same size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader


def build_test_loader(cfg):
    dataset_dicts = get_test_data_dicts(cfg.VISDRONE.TEST_JSON, cfg.VISDRONE.TEST_IMG_ROOT)

    dataset = DatasetFromList(dataset_dicts)
    mapper = Mapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


def trivial_batch_collator(batch):
    return batch
