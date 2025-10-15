"""
DeepLabV3+模型实现
使用预训练的ResNet作为backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+分割模型
    
    Args:
        n_classes (int): 分割类别数
        backbone (str): 骨干网络，可选 'resnet50' 或 'resnet101'
        pretrained (bool): 是否使用预训练权重
    """
    
    def __init__(self, n_classes=2, backbone='resnet50', pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        self.n_classes = n_classes
        self.backbone = backbone
        
        # 加载预训练模型
        if backbone == 'resnet50':
            self.model = deeplabv3_resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.model = deeplabv3_resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 修改分类器以适应目标类别数
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1)
        
        # 如果有辅助分类器，也需要修改
        if hasattr(self.model, 'aux_classifier'):
            self.model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1)
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # 前向传播
        output = self.model(x)['out']
        
        # 调整输出大小以匹配输入
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)
        
        return output


class ASPPConv(nn.Sequential):
    """ASPP中的卷积模块"""
    
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ASPP中的池化模块"""
    
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    空洞空间金字塔池化（Atrous Spatial Pyramid Pooling）
    """
    
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        # 空洞卷积
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # 全局平均池化
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # 融合所有特征
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


if __name__ == '__main__':
    # 测试模型
    model = DeepLabV3Plus(n_classes=2, backbone='resnet50', pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

