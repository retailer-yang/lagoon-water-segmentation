"""
日志工具
"""

import logging
import os
from datetime import datetime


def setup_logger(name='lagoon_segmentation', log_dir='logs', log_file=None):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        log_file: 日志文件名（可选，默认使用时间戳）
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'
    
    log_path = os.path.join(log_dir, log_file)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志系统已初始化，日志文件: {log_path}")
    
    return logger


class TensorboardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir='runs'):
        """
        初始化TensorBoard日志记录器
        
        Args:
            log_dir: 日志目录
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            print(f"TensorBoard日志目录: {log_dir}")
        except ImportError:
            print("警告: 未安装tensorboard，跳过TensorBoard日志记录")
            self.enabled = False
    
    def log_scalar(self, tag, value, step):
        """记录标量值"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, value_dict, step):
        """记录多个标量值"""
        if self.enabled:
            self.writer.add_scalars(tag, value_dict, step)
    
    def log_image(self, tag, image, step):
        """记录图像"""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def log_images(self, tag, images, step):
        """记录多张图像"""
        if self.enabled:
            self.writer.add_images(tag, images, step)
    
    def log_histogram(self, tag, values, step):
        """记录直方图"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """关闭记录器"""
        if self.enabled:
            self.writer.close()


if __name__ == '__main__':
    # 测试日志记录器
    logger = setup_logger('test')
    
    logger.debug('这是一条调试消息')
    logger.info('这是一条信息消息')
    logger.warning('这是一条警告消息')
    logger.error('这是一条错误消息')
    
    # 测试TensorBoard记录器
    tb_logger = TensorboardLogger('test_runs')
    tb_logger.log_scalar('test/loss', 0.5, 1)
    tb_logger.close()

