import os
import logging

def get_log_name(output_dir): # 根据实验次数自动生成新的日志文件名，避免重名
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = os.listdir(output_dir)
    log_list = [file for file in file_list if file.endswith('.log')]
    return output_dir + '/{}.log'.format(str(len(log_list)))

from logging import handlers
def get_logger(log_name): # 获取日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    rotate_handler = logging.handlers.TimedRotatingFileHandler(log_name, when='H', interval=1, backupCount=0)
    rotate_handler.setFormatter(formatter)
    logger.addHandler(rotate_handler)
    return logger

if __name__ == "__main__":
    log_name = get_log_name('./running_output') # 注意这个执行的时候是在utils文件夹下的
    logger = get_logger(log_name)
    loss_dict = {'total_loss': 1.2 , 'loss_cls': 0.4, 'loss_bbox_reg': 0.6, 'loss_query': 0.2}
    logger.info('total_loss:{:.4f}, loss_cls:{:.4f}, loss_bbox_reg:{:.4f}, loss_query:{:.4f}'.format(loss_dict['total_loss'], loss_dict['loss_cls'], loss_dict['loss_bbox_reg'], loss_dict['loss_query']))
    logger.info('test')