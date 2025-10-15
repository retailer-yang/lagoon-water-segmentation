"""
可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch


def visualize_prediction(image, mask, prediction, alpha=0.5, save_path=None):
    """
    可视化分割预测结果
    
    Args:
        image: 原始图像 (H, W, 3) 或 (3, H, W)
        mask: 真实掩码 (H, W)
        prediction: 预测掩码 (H, W) 或 (C, H, W)
        alpha: 叠加透明度
        save_path: 保存路径（可选）
    """
    # 转换tensor为numpy
    if torch.is_tensor(image):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        else:
            image = image.cpu().numpy()
    
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    if torch.is_tensor(prediction):
        if prediction.dim() == 3:
            prediction = torch.argmax(prediction, dim=0)
        prediction = prediction.cpu().numpy()
    
    # 归一化图像到[0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # 创建颜色映射
    colors = np.array([
        [0, 0, 0],      # 背景：黑色
        [0, 255, 255],  # 潟湖水域：青色
    ])
    
    # 创建彩色掩码
    mask_colored = colors[mask.astype(int)]
    pred_colored = colors[prediction.astype(int)]
    
    # 创建图形
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('原始图像', fontsize=12)
    axes[0].axis('off')
    
    # 真实掩码
    axes[1].imshow(image)
    axes[1].imshow(mask_colored / 255.0, alpha=alpha)
    axes[1].set_title('真实标注', fontsize=12)
    axes[1].axis('off')
    
    # 预测掩码
    axes[2].imshow(image)
    axes[2].imshow(pred_colored / 255.0, alpha=alpha)
    axes[2].set_title('预测结果', fontsize=12)
    axes[2].axis('off')
    
    # 对比（真实-绿色，预测-红色）
    overlay = image.copy()
    overlay = (overlay * 255).astype(np.uint8) if overlay.max() <= 1.0 else overlay.astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) if len(overlay.shape) == 3 else overlay
    
    # 绿色表示真实标注，红色表示预测
    comparison = np.zeros((*mask.shape, 3), dtype=np.uint8)
    comparison[mask == 1] = [0, 255, 0]  # 真实：绿色
    comparison[prediction == 1] = [255, 0, 0]  # 预测：红色
    comparison[(mask == 1) & (prediction == 1)] = [255, 255, 0]  # 重叠：黄色
    
    axes[3].imshow(image)
    axes[3].imshow(comparison / 255.0, alpha=alpha)
    axes[3].set_title('对比 (真实:绿, 预测:红, 重叠:黄)', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典，包含 'train_loss', 'val_loss', 'train_iou', 'val_iou' 等
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='训练损失', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('损失曲线', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU曲线
    if 'train_iou' in history:
        axes[1].plot(history['train_iou'], label='训练IoU', linewidth=2)
    if 'val_iou' in history:
        axes[1].plot(history['val_iou'], label='验证IoU', linewidth=2)
    if 'train_dice' in history:
        axes[1].plot(history['train_dice'], label='训练Dice', linewidth=2, linestyle='--')
    if 'val_dice' in history:
        axes[1].plot(history['val_dice'], label='验证Dice', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('评估指标曲线', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"训练历史曲线已保存到: {save_path}")
    
    plt.show()


def create_mask_overlay(image, mask, color=(0, 255, 255), alpha=0.5):
    """
    创建掩码叠加图像
    
    Args:
        image: 原始图像
        mask: 二值掩码
        color: 掩码颜色 (B, G, R)
        alpha: 透明度
    
    Returns:
        overlay: 叠加后的图像
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # 确保图像是uint8格式
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # 创建彩色掩码
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = color
    
    # 叠加
    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    
    return overlay


def save_prediction_grid(images, masks, predictions, save_path, n_samples=4):
    """
    保存预测结果网格图
    
    Args:
        images: 图像列表
        masks: 真实掩码列表
        predictions: 预测掩码列表
        save_path: 保存路径
        n_samples: 显示样本数
    """
    n_samples = min(n_samples, len(images))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # 原始图像
        img = images[i]
        if torch.is_tensor(img):
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0).cpu().numpy()
            else:
                img = img.cpu().numpy()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'样本 {i+1} - 原始图像')
        axes[i, 0].axis('off')
        
        # 真实掩码
        mask = masks[i]
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('真实标注')
        axes[i, 1].axis('off')
        
        # 预测掩码
        pred = predictions[i]
        if torch.is_tensor(pred):
            if pred.dim() == 3:
                pred = torch.argmax(pred, dim=0)
            pred = pred.cpu().numpy()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('预测结果')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"预测结果网格图已保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    # 测试可视化
    image = np.random.rand(512, 512, 3)
    mask = np.random.randint(0, 2, (512, 512))
    prediction = np.random.randint(0, 2, (512, 512))
    
    visualize_prediction(image, mask, prediction)

