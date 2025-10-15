"""
潟湖水域分割数据集类
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class LagoonDataset(Dataset):
    """
    潟湖水域分割数据集
    
    Args:
        image_dir (str): 图像目录路径
        mask_dir (str): 掩码目录路径
        transform (callable, optional): 图像变换
        augmentation (callable, optional): 数据增强
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, augmentation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augmentation = augmentation
        
        # 获取所有图像文件
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
        
        if len(self.images) == 0:
            print(f"警告: {image_dir} 目录中没有找到图像文件")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 读取图像（支持多通道）
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载掩码
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # 如果掩码不存在，创建空掩码
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 数据增强
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_name
        }


class LagoonTestDataset(Dataset):
    """
    测试数据集（仅包含图像，无掩码）
    """
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return {
            'image': image,
            'filename': img_name
        }

