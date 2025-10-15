"""
训练脚本
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data.dataset import LagoonDataset
from src.data.preprocessing import get_training_augmentation, get_validation_augmentation
from src.models.model_factory import create_model, count_parameters
from src.utils.metrics import calculate_metrics, AverageMeter
from src.utils.visualization import plot_training_history
from src.utils.logger import setup_logger, TensorboardLogger


class Trainer:
    """训练器类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name='training',
            log_dir=str(self.output_dir / 'logs')
        )
        self.tb_logger = TensorboardLogger(
            log_dir=str(self.output_dir / 'tensorboard')
        )
        
        # 构建模型
        self.logger.info("构建模型...")
        self.model = create_model(
            model_name=config['model']['name'],
            n_classes=config['model']['n_classes'],
            **config['model'].get('params', {})
        )
        self.model = self.model.to(self.device)
        
        total_params, trainable_params = count_parameters(self.model)
        self.logger.info(f"模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")
        
        # 构建数据加载器
        self.logger.info("加载数据...")
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # 定义损失函数
        self.criterion = self._build_criterion()
        
        # 定义优化器
        self.optimizer = self._build_optimizer()
        
        # 定义学习率调度器
        self.scheduler = self._build_scheduler()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': []
        }
        
        self.best_val_iou = 0.0
        self.start_epoch = 0
        
        # 如果有检查点，加载它
        if config.get('resume'):
            self._load_checkpoint(config['resume'])
    
    def _build_dataloaders(self):
        """构建数据加载器"""
        train_dataset = LagoonDataset(
            image_dir=self.config['data']['train_images'],
            mask_dir=self.config['data']['train_masks'],
            augmentation=get_training_augmentation(
                image_size=self.config['data']['image_size']
            )
        )
        
        val_dataset = LagoonDataset(
            image_dir=self.config['data']['val_images'],
            mask_dir=self.config['data']['val_masks'],
            augmentation=get_validation_augmentation(
                image_size=self.config['data']['image_size']
            )
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        self.logger.info(f"训练集: {len(train_dataset)} 样本")
        self.logger.info(f"验证集: {len(val_dataset)} 样本")
        
        return train_loader, val_loader
    
    def _build_criterion(self):
        """构建损失函数"""
        criterion_name = self.config['training']['criterion']
        
        if criterion_name == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif criterion_name == 'DiceLoss':
            criterion = DiceLoss()
        elif criterion_name == 'CombinedLoss':
            criterion = CombinedLoss()
        else:
            raise ValueError(f"不支持的损失函数: {criterion_name}")
        
        return criterion
    
    def _build_optimizer(self):
        """构建优化器"""
        optimizer_name = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 1e-4)
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        return optimizer
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        scheduler_name = self.config['training'].get('scheduler', 'ReduceLROnPlateau')
        
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["epochs"]} [训练]')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                metrics = calculate_metrics(
                    outputs,
                    masks,
                    n_classes=self.config['model']['n_classes']
                )
            
            # 更新统计
            loss_meter.update(loss.item(), images.size(0))
            iou_meter.update(metrics['mean_iou'], images.size(0))
            dice_meter.update(metrics['mean_dice'], images.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'iou': f'{iou_meter.avg:.4f}',
                'dice': f'{dice_meter.avg:.4f}'
            })
        
        return loss_meter.avg, iou_meter.avg, dice_meter.avg
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.config["training"]["epochs"]} [验证]')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # 计算指标
                metrics = calculate_metrics(
                    outputs,
                    masks,
                    n_classes=self.config['model']['n_classes']
                )
                
                # 更新统计
                loss_meter.update(loss.item(), images.size(0))
                iou_meter.update(metrics['mean_iou'], images.size(0))
                dice_meter.update(metrics['mean_dice'], images.size(0))
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'iou': f'{iou_meter.avg:.4f}',
                    'dice': f'{dice_meter.avg:.4f}'
                })
        
        return loss_meter.avg, iou_meter.avg, dice_meter.avg
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_iou': self.best_val_iou,
            'history': self.history,
            'config': self.config
        }
        
        # 保存最新的检查点
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_iou = checkpoint['best_val_iou']
        self.history = checkpoint['history']
    
    def train(self):
        """训练主循环"""
        self.logger.info("开始训练...")
        self.logger.info(f"设备: {self.device}")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # 训练
            train_loss, train_iou, train_dice = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_iou, val_dice = self.validate(epoch)
            
            # 更新历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            
            # 记录到TensorBoard
            self.tb_logger.log_scalar('Loss/train', train_loss, epoch)
            self.tb_logger.log_scalar('Loss/val', val_loss, epoch)
            self.tb_logger.log_scalar('IoU/train', train_iou, epoch)
            self.tb_logger.log_scalar('IoU/val', val_iou, epoch)
            self.tb_logger.log_scalar('Dice/train', train_dice, epoch)
            self.tb_logger.log_scalar('Dice/val', val_dice, epoch)
            
            # 日志记录
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, Train IoU={train_iou:.4f}, "
                f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
            )
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_iou)
                else:
                    self.scheduler.step()
            
            # 保存检查点
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
            
            self.save_checkpoint(epoch, is_best)
        
        # 训练完成
        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证IoU: {self.best_val_iou:.4f}")
        
        # 绘制训练历史
        plot_training_history(
            self.history,
            save_path=str(self.output_dir / 'training_history.png')
        )
        
        self.tb_logger.close()


class DiceLoss(nn.Module):
    """Dice损失函数"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """组合损失：交叉熵 + Dice"""
    
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def main():
    parser = argparse.ArgumentParser(description='训练潟湖水域分割模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.resume:
        config['resume'] = args.resume
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

