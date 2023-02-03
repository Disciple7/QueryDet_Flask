import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))

import cv2
import json
from tqdm import tqdm



def get_save_path(img_path, index):
    name = img_path.split('.')[0]
    return name + '_' + str(index) + '.jpg'

def crop_and_save_image(img_root, img_path, new_img_root):
    img = cv2.imread(os.path.join(img_root,img_path))
    h, w, c = img.shape

    _y = h // 2
    _x = w // 2
    
    img0 = img[:_y, :_x, :]
    img1 = img[:_y, _x:, :]
    img2 = img[_y:, :_x, :]
    img3 = img[_y:, _x:, :]

    cv2.imwrite(os.path.join(new_img_root, get_save_path(img_path, 0)), img0)
    cv2.imwrite(os.path.join(new_img_root, get_save_path(img_path, 1)), img1)
    cv2.imwrite(os.path.join(new_img_root, get_save_path(img_path, 2)), img2)
    cv2.imwrite(os.path.join(new_img_root, get_save_path(img_path, 3)), img3)

    return h, w, _y, _x


def copy_image(img_root, img_path, new_img_root):
    img = cv2.imread(os.path.join(img_root,img_path))
    h, w, c = img.shape
    cv2.imwrite(os.path.join(new_img_root, img_path), img)
    return h, w


def get_new_label(label, img_path, cy, cx, id, img_id_base):
    if label['class'] == 0 or label['ignore']:
        return None

    x, y, w, h = label['bbox']
    
    if x < cx and y < cy:
        nx = x
        ny = y
        nw = min(x+w, cx) - x
        nh = min(y+h, cy) - y
        img_id = img_id_base
    elif x < cx and y >= cy:
        nx = x
        ny = y - cy
        nw = min(x+w, cx) - x
        nh = h
        img_id = img_id_base + 2
    elif x >= cx and y < cy:
        nx = x - cx
        ny = y
        nw = w
        nh = min(y+h, cy) - y
        img_id = img_id_base + 1
    else:
        nx = x - cx
        ny = y - cy
        nw = w
        nh = h
        img_id = img_id_base + 3
    
    new_label = {'category_id': label['class'], 'id': id, 'iscrowd':0, 'image_id':img_id, 'area':nw*nh, 'segmentation':[], 'bbox':[nx,ny,nw,nh]}
    return new_label


def label_to_coco(label, id, img_id):
    x, y, w, h = label['bbox']
    new_label = {'category_id': label['class'], 'id': id, 'iscrowd':0, 'image_id':img_id, 'area':w*h, 'segmentation':[], 'bbox':[x,y,w,h]}
    return new_label


def make_json(images, annotations, new_label_json):
    ann_dict = {}
    ann_dict['categories'] = [
        {'supercategory': 'things', 'id': 1, 'name': 'people'},
        {'supercategory': 'things', 'id': 2, 'name': 'bicycle'},
        {'supercategory': 'things', 'id': 3, 'name': 'car'},
        {'supercategory': 'things', 'id': 4, 'name': 'van'},
        {'supercategory': 'things', 'id': 5, 'name': 'truck'},
        {'supercategory': 'things', 'id': 6, 'name': 'tricycle'},
        {'supercategory': 'things', 'id': 7, 'name': 'awning-tricycle'},
        {'supercategory': 'things', 'id': 8, 'name': 'bus'},
        {'supercategory': 'things', 'id': 9, 'name': 'motor'}
    ]
    ann_dict['images'] = images
    ann_dict['annotations'] = annotations
    with open(new_label_json, 'w') as outfile:
        json.dump(ann_dict, outfile)


def make_new_train_set(img_root, label_root, new_img_root, new_label_json):
    all_labels = read_all_labels(label_root)

    annotations = []
    images = []
    ann_id = 0
    img_id = 0
    for filename, labels in tqdm(all_labels.items()):
        img_path = filename.replace('txt', 'jpg')
        h, w, cy, cx = crop_and_save_image(img_root, img_path, new_img_root)
        
        images.append({'file_name': get_save_path(img_path, 0), 'height': cy, 'width': cx, 'id': img_id})
        images.append({'file_name': get_save_path(img_path, 1), 'height': cy, 'width': w-cx, 'id': img_id+1})
        images.append({'file_name': get_save_path(img_path, 2), 'height': h-cy, 'width': cx, 'id':img_id+2})
        images.append({'file_name': get_save_path(img_path, 3), 'height': h-cy, 'width': w-cx, 'id':img_id+3})

        for label in labels:
            new_label = get_new_label(label, img_path, cy, cx, ann_id, img_id)
            if new_label != None:
                ann_id += 1
                annotations.append(new_label)
        img_id += 4
    make_json(images, annotations, new_label_json)


def make_new_test_set(img_root, label_root, new_img_root, new_label_json):
    all_labels = read_all_labels(label_root)
    annotations = []
    images = []
    ann_id = 0
    img_id = 0

    for filename, labels in tqdm(all_labels.items()):
        img_path = filename.replace('txt', 'jpg')
        h, w = copy_image(img_root, img_path, new_img_root)
        images.append({'file_name': img_path, 'height': h, 'width': w, 'id': img_id})
        for label in labels:
            coco_label = label_to_coco(label, ann_id, img_id)
            if coco_label != None:
                ann_id += 1
                annotations.append(coco_label)
        img_id += 1
    
    make_json(images, annotations, new_label_json)

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

if __name__ == '__main__':
    '''
    Training
    '''
    img_root = './VisDrone2018-DET-train/images'
    label_root = './VisDrone2018-DET-train/annotations'
    new_img_root = './visdrone_coco_train/images'
    new_label_json = './visdrone_coco_train/label.json'
    make_new_train_set(img_root, label_root, new_img_root, new_label_json)

    '''
    Validation
    '''
    img_root = './VisDrone2018-DET-val/images'
    label_root = './VisDrone2018-DET-val/annotations'
    new_img_root = './visdrone_coco_val/images'
    new_label_json = './visdrone_coco_val/label.json'
    make_new_test_set(img_root, label_root, new_img_root, new_label_json)


    '''
    Test
    '''
    img_root = './VisDrone2018-DET-test-challenge/images'
    # label_root = './VisDrone2018-DET-test-challenge/annotations'
    new_img_root = './visdrone_coco_test/images'
    new_label_json = './visdrone_coco_test/label.json'
    make_new_test_set(img_root, label_root, new_img_root, new_label_json)

    

    

