import os
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_coco(gt_json_file, det_json_file):
    coco_gt = COCO(gt_json_file)
    # coco_results, coco_rebuilt_gt = build_coco_results(pred_list, thres_fix=True, thres=0.4)
    coco_results = COCO(det_json_file)
    for ann in coco_results.loadAnns(coco_results.getAnnIds()):
        ann['category_id'] += 1
    coco_eval = COCOeval(coco_gt, coco_results, 'bbox')  # 必须要gt和result的ImgID一模一样，代码内部是根据ImgID来匹配的
    # coco_eval.params.imgIds = coco_rebuilt_gt.getImgIds()
    with open(det_json_file + '_cocoeval.txt', 'w') as fp:
        sys.stdout = fp
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        sys.stdout = sys.__stdout__
    return None


def batch_eval():
    for filename in os.listdir('.'):
        if filename.endswith('result.json'):
            eval_coco('./visdrone_coco_val/label.json', filename)


if __name__ == '__main__':
    # eval_coco('./visdrone_coco_val/label.json', '110000_model_result.json')
    batch_eval()