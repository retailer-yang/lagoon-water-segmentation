"""
评估脚本
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
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.dataset import LagoonDataset
from src.data.preprocessing import get_validation_augmentation
from src.models.model_factory import load_model
from src.utils.metrics import calculate_metrics, AverageMeter


def evaluate(model, dataloader, device, n_classes=2):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        n_classes: 类别数
    
    Returns:
        results: 评估结果字典
    """
    model.eval()
    
    # 初始化指标记录器
    acc_meter = AverageMeter()
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()
    
    all_predictions = []
    all_targets = []
    
    print("开始评估...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='评估中'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算指标
            metrics = calculate_metrics(outputs, masks, n_classes=n_classes)
            
            # 更新统计
            acc_meter.update(metrics['pixel_accuracy'], images.size(0))
            iou_meter.update(metrics['mean_iou'], images.size(0))
            dice_meter.update(metrics['mean_dice'], images.size(0))
            precision_meter.update(metrics['precision'], images.size(0))
            recall_meter.update(metrics['recall'], images.size(0))
            f1_meter.update(metrics['f1_score'], images.size(0))
            
            # 保存预测和真实值
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # 汇总结果
    results = {
        'pixel_accuracy': acc_meter.avg,
        'mean_iou': iou_meter.avg,
        'mean_dice': dice_meter.avg,
        'precision': precision_meter.avg,
        'recall': recall_meter.avg,
        'f1_score': f1_meter.avg,
        'predictions': np.concatenate(all_predictions, axis=0),
        'targets': np.concatenate(all_targets, axis=0)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估潟湖水域分割模型')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, required=True, help='测试数据目录')
    parser.add_argument('--mask_dir', type=str, required=True, help='测试掩码目录')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--output', type=str, help='输出结果保存路径')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = load_model(
        model_path=args.model,
        model_name=config['model']['name'],
        n_classes=config['model']['n_classes'],
        device=str(device),
        **config['model'].get('params', {})
    )
    
    # 加载数据
    print("加载测试数据...")
    test_dataset = LagoonDataset(
        image_dir=args.data_dir,
        mask_dir=args.mask_dir,
        augmentation=get_validation_augmentation(
            image_size=config['data']['image_size']
        )
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 评估
    results = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        n_classes=config['model']['n_classes']
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    print(f"像素准确率 (Pixel Accuracy): {results['pixel_accuracy']:.4f}")
    print(f"平均IoU (Mean IoU):          {results['mean_iou']:.4f}")
    print(f"平均Dice (Mean Dice):         {results['mean_dice']:.4f}")
    print(f"精确率 (Precision):           {results['precision']:.4f}")
    print(f"召回率 (Recall):              {results['recall']:.4f}")
    print(f"F1分数 (F1 Score):            {results['f1_score']:.4f}")
    print("="*50)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存评估指标
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("潟湖水域分割模型评估结果\n")
            f.write("="*50 + "\n")
            f.write(f"模型: {args.model}\n")
            f.write(f"测试数据: {args.data_dir}\n")
            f.write(f"样本数: {len(test_dataset)}\n")
            f.write("="*50 + "\n")
            f.write(f"像素准确率: {results['pixel_accuracy']:.4f}\n")
            f.write(f"平均IoU: {results['mean_iou']:.4f}\n")
            f.write(f"平均Dice: {results['mean_dice']:.4f}\n")
            f.write(f"精确率: {results['precision']:.4f}\n")
            f.write(f"召回率: {results['recall']:.4f}\n")
            f.write(f"F1分数: {results['f1_score']:.4f}\n")
        
        print(f"\n评估结果已保存到: {output_path}")


if __name__ == '__main__':
    main()

