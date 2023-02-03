import numpy as np
import torch.optim
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.config import CfgNode

from detectron2.checkpoint import DetectionCheckpointer

from modeling.main_model import RetinaNetQueryDet, add_querydet_config

import matplotlib.pyplot as plt
from dataloader import build_train_loader

import detectron2

import cv2

from fvcore.common.param_scheduler import MultiStepParamScheduler

from utils.record import get_log_name, get_logger

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


def show_batch_gt_box(coco, batch_data):
    for data in batch_data:
        img = np.ascontiguousarray(np.transpose(data[0]['image'].numpy(), [1, 2, 0]))
        anns = data[0]['instances']._fields['gt_boxes']
        for ann in anns:
            x1, y1, x2, y2 = ann.T.numpy()
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    cfg = get_cfg()  # cfg默认配置来自D:\Code\coding-py\detectron2\detectron2\config\defaults.py
    add_querydet_config(cfg)
    cfg.merge_from_file('./config/train.yaml')
    cfg.freeze()
    default_setup(cfg, None)

    # 构建数据集读取器
    dataloader = build_train_loader(cfg)
    data_loader_obj = iter(dataloader)
    batch_data = [next(data_loader_obj) for _ in range(5)]

    # coco = COCO(cfg.VISDRONE.TRAIN_JSON)
    # show_batch_gt_box(coco, batch_data) # 抽样了一下，虽然标注有点差，但是输入应该没什么问题。

    # 构建模型
    model = RetinaNetQueryDet(cfg)
    device = torch.device('cuda:1')
    # 构建优化器
    optimizer = torch.optim.SGD([{'params':model.parameters(), 'initial_lr':1e-3}],
                                lr=cfg.SOLVER.BASE_LR,
                                momentum=cfg.SOLVER.MOMENTUM)
    # 学习率策略
    steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
    sched = MultiStepParamScheduler(
        values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
        milestones=steps,
        num_updates=cfg.SOLVER.MAX_ITER,
    )
    sched = detectron2.solver.WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
    )
    # 模型恢复训练相关设置
    last_iter = -1 # 设置上次保存点。如果从头训练，设置为-1。后续可以加到配置文件中。
    scheduler = detectron2.solver.LRMultiplier(
        optimizer,
        multiplier=sched,
        max_iter=cfg.SOLVER.MAX_ITER,
        last_iter=last_iter if last_iter > 0 else -1,
        # verbose=True
    )
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    if last_iter == -1:
        start_iter = 0
        checkpointer.load(cfg.MODEL.WEIGHTS)
    else:
        start_iter = last_iter
        checkpointer.resume_or_load(cfg.OUTPUT_DIR + '/{}.pth'.format(str(last_iter)))
        # checkpointer.resume_or_load(cfg.MODEL.WEIGHTS)

    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    max_iter = cfg.SOLVER.MAX_ITER
    # with EventStorage(start_iter=start_iter) as storage:
    record_name = get_log_name(cfg.OUTPUT_DIR)
    logger = get_logger(record_name)
    for itr in range(start_iter, max_iter):
        data = next(data_loader_obj)
        with torch.amp.autocast(device_type='cuda'):
            loss_dict = model(data)  # 这模型forward里面写这么多东西
        # 正向跑完一次以后模型就用了6.8G的显存了，一算反向就爆炸了
        losses = sum(loss for loss in loss_dict.values())
        logger.info('iter:{:d}, total_loss:{:.4f}, loss_cls:{:.4f}, loss_box_reg:{:.4f}, loss_query:{:.4f}'
                    .format(itr, float(losses), float(loss_dict['loss_cls']), float(loss_dict['loss_box_reg']), float(loss_dict['loss_query'])))
        scheduler.step()
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            scaler.scale(losses).backward()
            # losses.backward()
            # optimizer.step()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=128.)
        scaler.step(optimizer)
        scaler.update()

        # 梯度爆炸了。学习率过大？数据不正确？没有归一化？梯度裁剪？
        # 目前看来是学习率过大的问题，lr=0.01时爆炸，0.001时暂时没问题。
        if itr % 10000 == 9999:
            torch.save(model.state_dict(), cfg.OUTPUT_DIR + '/{}.pth'.format(itr))
            logger.info('save model to {}'.format(cfg.OUTPUT_DIR + '/{}.pth'.format(itr)))