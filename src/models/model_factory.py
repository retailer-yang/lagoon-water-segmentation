"""
模型工厂
用于创建和管理不同的模型
"""

import torch
from .unet import UNet
from .deeplabv3 import DeepLabV3Plus


def create_model(model_name, n_classes=2, **kwargs):
    """
    创建指定的模型
    
    Args:
        model_name (str): 模型名称 ('unet', 'deeplabv3')
        n_classes (int): 分割类别数
        **kwargs: 其他模型参数
    
    Returns:
        model: PyTorch模型
    """
    model_name = model_name.lower()
    
    if model_name == 'unet':
        model = UNet(
            n_channels=kwargs.get('n_channels', 3),
            n_classes=n_classes,
            bilinear=kwargs.get('bilinear', True)
        )
    
    elif model_name == 'deeplabv3':
        model = DeepLabV3Plus(
            n_classes=n_classes,
            backbone=kwargs.get('backbone', 'resnet50'),
            pretrained=kwargs.get('pretrained', True)
        )
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model


def load_model(model_path, model_name, n_classes=2, device='cpu', **kwargs):
    """
    加载已保存的模型
    
    Args:
        model_path (str): 模型权重路径
        model_name (str): 模型名称
        n_classes (int): 分割类别数
        device (str): 设备 ('cpu' 或 'cuda')
        **kwargs: 其他模型参数
    
    Returns:
        model: 加载权重后的模型
    """
    model = create_model(model_name, n_classes, **kwargs)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def count_parameters(model):
    """
    计算模型参数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


if __name__ == '__main__':
    # 测试模型创建
    print("测试 U-Net:")
    unet = create_model('unet', n_classes=2)
    total, trainable = count_parameters(unet)
    print(f"总参数: {total:,}, 可训练参数: {trainable:,}")
    
    print("\n测试 DeepLabV3+:")
    deeplab = create_model('deeplabv3', n_classes=2, pretrained=False)
    total, trainable = count_parameters(deeplab)
    print(f"总参数: {total:,}, 可训练参数: {trainable:,}")

