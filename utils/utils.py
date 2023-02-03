import torch
import os

from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat
from detectron2.layers import cat

def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta

def permute_all_to_NHWA_K_not_concat(box_cls, box_delta, num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes).reshape(-1, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4).reshape(-1, 4) for x in box_delta]
    return box_cls_flattened, box_delta_flattened

from detectron2.structures import Boxes
def get_box_scales(boxes: Boxes):
    return torch.sqrt((boxes.tensor[:, 2] - boxes.tensor[:, 0]) * (boxes.tensor[:, 3] - boxes.tensor[:, 1]))


def get_anchor_center_min_dis(box_centers: torch.Tensor, anchor_centers: torch.Tensor):
    """
    Args:
        box_centers: [N, 2]
        anchor_centers: [M, 2]
    Returns:

    """
    N, _ = box_centers.size()
    M, _ = anchor_centers.size()
    if N == 0:
        return torch.ones_like(anchor_centers)[:, 0] * 99999, (torch.zeros_like(anchor_centers)[:, 0]).long()
    acenters = anchor_centers.view(-1, 1, 2)
    acenters = acenters.repeat(1, N, 1)
    bcenters = box_centers.view(1, -1, 2)
    bcenters = bcenters.repeat(M, 1, 1)

    dis = torch.sqrt(torch.sum((acenters - bcenters) ** 2, dim=2))

    mindis, minind = torch.min(input=dis, dim=1)

    return mindis, minind


def read_label_txt(txt_file):
    f = open(txt_file, 'r')
    lines = f.readlines()

    labels = []
    for line in lines:
        line = line.strip().split(',')

        x, y, w, h, not_ignore, cate, trun, occ = line[:8]

        labels.append(
            {'bbox': (int(x),int(y),int(w),int(h)),
             'ignore': 0 if int(not_ignore) else 1,
             'class': int(cate),
             'truncate': int(trun),
             'occlusion': int(occ)}
        )
    return labels


def read_all_labels(ann_root):
    ann_list = os.listdir(ann_root)
    all_labels = {}
    for ann_file in ann_list:
        if not ann_file.endswith('txt'):
            continue
        ann_labels = read_label_txt(os.path.join(ann_root, ann_file))
        all_labels[ann_file] = ann_labels
    return all_labels