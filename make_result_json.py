import json
import torch

def make_result_json(pred_list, thres_fix = False, thres = 0.4, model_name = 'default'):
    images = []
    annotations = []
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
            images.append({
                'file_name': data['file_name'],
                'height': data['height'],
                'width': data['width'],
                'id': data['image_id']
            })
            for ann, score, pred_class in zip(anns, scores, pred_classes):
                if thres_fix and float(score) < thres:
                    continue
                x1, y1, x2, y2 = torch.Tensor.cpu(ann.detach().T).numpy()
                x1 = max(0, x1) * x_scale_factor
                y1 = max(0, y1) * y_scale_factor
                x2 = min(data['width'], x2) * x_scale_factor
                y2 = min(data['height'], y2) * y_scale_factor
                annotations.append({
                    'category_id': int(pred_class),
                    'id': len(annotations), # annotation的id是全局的，每个都不一样。通过image_id区分不同图片的annotation
                    'image_id': data['image_id'],
                    'area': (x2 - x1) * (y2 - y1),
                    'bbox': [x1, y1, int(abs(x2 - x1)), int(abs(y2 - y1))],
                    'score': float(score)
                })
    ann_dict = {}
    ann_dict['categories'] = [
        {'supercategory': 'things', 'id': 0, 'name': 'pedestrian'},
        {'supercategory': 'things', 'id': 1, 'name': 'people'},
        {'supercategory': 'things', 'id': 2, 'name': 'bicycle'},
        {'supercategory': 'things', 'id': 3, 'name': 'car'},
        {'supercategory': 'things', 'id': 4, 'name': 'van'},
        {'supercategory': 'things', 'id': 5, 'name': 'truck'},
        {'supercategory': 'things', 'id': 6, 'name': 'tricycle'},
        {'supercategory': 'things', 'id': 7, 'name': 'awning-tricycle'},
        {'supercategory': 'things', 'id': 8, 'name': 'bus'},
        {'supercategory': 'things', 'id': 9, 'name': 'motor'},
        {'supercategory': 'things', 'id': 10, 'name': 'others'},
        {'supercategory': 'things', 'id': 11, 'name': '???'}
    ]
    ann_dict['images'] = images
    ann_dict['annotations'] = annotations
    if thres_fix :
        save_name = '{}_model_result_thres_{}.json'.format(model_name, str(thres))
    else:
        save_name = '{}_model_result.json'.format(model_name)
    with open(save_name, 'w') as outfile:
        json.dump(ann_dict, outfile)
