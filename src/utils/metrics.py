"""
评估指标
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def IoU(pred, target, n_classes=2, eps=1e-7):
    """
    计算交并比（Intersection over Union）
    
    Args:
        pred: 预测结果 (B, H, W) 或 (B, C, H, W)
        target: 真实标签 (B, H, W)
        n_classes: 类别数
        eps: 防止除零的小数
    
    Returns:
        iou: 各类别的IoU
    """
    # 如果pred是概率图，转换为类别预测
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    ious = []
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    
    for cls in range(n_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = float('nan')  # 该类别不存在
        else:
            iou = (intersection + eps) / (union + eps)
        
        ious.append(iou)
    
    return ious


def DiceCoefficient(pred, target, n_classes=2, eps=1e-7):
    """
    计算Dice系数
    
    Args:
        pred: 预测结果 (B, H, W) 或 (B, C, H, W)
        target: 真实标签 (B, H, W)
        n_classes: 类别数
        eps: 防止除零的小数
    
    Returns:
        dice: 各类别的Dice系数
    """
    # 如果pred是概率图，转换为类别预测
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    dice_scores = []
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    
    for cls in range(n_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        
        if pred_cls.sum() + target_cls.sum() == 0:
            dice = float('nan')
        else:
            dice = (2.0 * intersection + eps) / (pred_cls.sum() + target_cls.sum() + eps)
        
        dice_scores.append(dice)
    
    return dice_scores


def pixel_accuracy(pred, target):
    """
    计算像素准确率
    
    Args:
        pred: 预测结果 (B, H, W) 或 (B, C, H, W)
        target: 真实标签 (B, H, W)
    
    Returns:
        accuracy: 像素准确率
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    
    accuracy = accuracy_score(target, pred)
    return accuracy


def calculate_metrics(pred, target, n_classes=2):
    """
    计算所有评估指标
    
    Args:
        pred: 预测结果 (B, H, W) 或 (B, C, H, W)
        target: 真实标签 (B, H, W)
        n_classes: 类别数
    
    Returns:
        metrics: 包含各种指标的字典
    """
    # 如果pred是概率图，转换为类别预测
    if pred.dim() == 4:
        pred_labels = torch.argmax(pred, dim=1)
    else:
        pred_labels = pred
    
    pred_np = pred_labels.view(-1).cpu().numpy()
    target_np = target.view(-1).cpu().numpy()
    
    # 计算各种指标
    iou = IoU(pred, target, n_classes)
    dice = DiceCoefficient(pred, target, n_classes)
    acc = pixel_accuracy(pred, target)
    
    # 计算精确率、召回率、F1分数（仅对二分类）
    if n_classes == 2:
        precision = precision_score(target_np, pred_np, average='binary', zero_division=0)
        recall = recall_score(target_np, pred_np, average='binary', zero_division=0)
        f1 = f1_score(target_np, pred_np, average='binary', zero_division=0)
    else:
        precision = precision_score(target_np, pred_np, average='macro', zero_division=0)
        recall = recall_score(target_np, pred_np, average='macro', zero_division=0)
        f1 = f1_score(target_np, pred_np, average='macro', zero_division=0)
    
    metrics = {
        'pixel_accuracy': acc,
        'mean_iou': np.nanmean(iou),
        'iou_per_class': iou,
        'mean_dice': np.nanmean(dice),
        'dice_per_class': dice,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # 测试指标计算
    pred = torch.randint(0, 2, (4, 512, 512))
    target = torch.randint(0, 2, (4, 512, 512))
    
    metrics = calculate_metrics(pred, target, n_classes=2)
    
    print("评估指标:")
    for key, value in metrics.items():
        if isinstance(value, list):
            print(f"{key}: {[f'{v:.4f}' if not np.isnan(v) else 'NaN' for v in value]}")
        else:
            print(f"{key}: {value:.4f}")

