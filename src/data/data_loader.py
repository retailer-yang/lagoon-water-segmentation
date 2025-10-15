"""
数据加载器
"""

import os
from torch.utils.data import DataLoader, random_split
from .dataset import LagoonDataset, LagoonTestDataset
from .preprocessing import get_training_augmentation, get_validation_augmentation


def get_data_loaders(
    image_dir,
    mask_dir,
    batch_size=8,
    num_workers=4,
    image_size=512,
    val_split=0.2,
    seed=42
):
    """
    创建训练和验证数据加载器
    
    Args:
        image_dir: 图像目录路径
        mask_dir: 掩码目录路径
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        image_size: 图像尺寸
        val_split: 验证集比例
        seed: 随机种子
    
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 创建完整数据集
    full_dataset = LagoonDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None,
        augmentation=get_training_augmentation(image_size)
    )
    
    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 为验证集设置不同的增强（仅预处理，无数据增强）
    val_dataset.dataset.augmentation = get_validation_augmentation(image_size)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_test_loader(
    image_dir,
    batch_size=8,
    num_workers=4,
    image_size=512
):
    """
    创建测试数据加载器
    
    Args:
        image_dir: 图像目录路径
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        image_size: 图像尺寸
    
    Returns:
        test_loader: 测试数据加载器
    """
    test_dataset = LagoonTestDataset(
        image_dir=image_dir,
        transform=get_validation_augmentation(image_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader


# 导入torch（在函数外部）
import torch

