"""
数据预处理模块
"""

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def preprocess_image(image, target_size=(512, 512)):
    """
    预处理图像
    
    Args:
        image: 输入图像（numpy array 或 PIL Image）
        target_size: 目标尺寸 (height, width)
    
    Returns:
        preprocessed_image: 预处理后的图像
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 调整大小
    image = cv2.resize(image, (target_size[1], target_size[0]), 
                      interpolation=cv2.INTER_LINEAR)
    
    # 归一化到 [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image


def get_training_augmentation(image_size=512):
    """
    获取训练时的数据增强
    
    Args:
        image_size: 图像尺寸
    
    Returns:
        albumentations.Compose: 数据增强管道
    """
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),
        
        # 颜色增强
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.ColorJitter(p=1.0),
        ], p=0.5),
        
        # 模糊和噪声
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
        
        # 归一化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    return train_transform


def get_validation_augmentation(image_size=512):
    """
    获取验证时的数据增强（仅包含必要的预处理）
    
    Args:
        image_size: 图像尺寸
    
    Returns:
        albumentations.Compose: 数据增强管道
    """
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    return val_transform


def augment_image(image, mask=None):
    """
    简单的图像增强函数
    
    Args:
        image: 输入图像
        mask: 对应的掩码（可选）
    
    Returns:
        augmented_image, augmented_mask (if mask provided)
    """
    transform = get_training_augmentation()
    
    if mask is not None:
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    else:
        augmented = transform(image=image)
        return augmented['image']


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反归一化图像用于可视化
    
    Args:
        tensor: 归一化的图像tensor
        mean: 归一化使用的均值
        std: 归一化使用的标准差
    
    Returns:
        denormalized_image: 反归一化后的图像
    """
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    
    image = tensor.cpu().numpy() * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    
    return image

