"""
推理脚本
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
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models.model_factory import load_model
from src.data.preprocessing import get_validation_augmentation
from src.utils.visualization import visualize_prediction, save_prediction_grid


class Predictor:
    """预测器类"""
    
    def __init__(self, model_path, config_path, device='cuda'):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重路径
            config_path: 配置文件路径
            device: 设备 ('cuda' 或 'cpu')
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = load_model(
            model_path=model_path,
            model_name=self.config['model']['name'],
            n_classes=self.config['model']['n_classes'],
            device=str(self.device),
            **self.config['model'].get('params', {})
        )
        
        # 数据预处理
        self.transform = get_validation_augmentation(
            image_size=self.config['data']['image_size']
        )
        
        print("预测器初始化完成")
    
    def predict_image(self, image_path):
        """
        预测单张图像
        
        Args:
            image_path: 图像路径
        
        Returns:
            prediction: 预测掩码
            probability: 预测概率图
        """
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # 预处理
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.softmax(output, dim=1)
            prediction = torch.argmax(probability, dim=1)
        
        # 转换为numpy
        prediction = prediction.squeeze(0).cpu().numpy()
        probability = probability.squeeze(0).cpu().numpy()
        
        # 调整回原始尺寸
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        probability_resized = []
        for i in range(probability.shape[0]):
            prob = cv2.resize(
                probability[i],
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            probability_resized.append(prob)
        probability = np.array(probability_resized)
        
        return prediction, probability
    
    def predict_directory(self, input_dir, output_dir, visualize=True, save_mask=True):
        """
        预测整个目录的图像
        
        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            visualize: 是否保存可视化结果
            save_mask: 是否保存掩码
        """
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_mask:
            mask_dir = output_dir / 'masks'
            mask_dir.mkdir(exist_ok=True)
        
        if visualize:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend(Path(input_dir).glob(ext))
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 预测每张图像
        for image_path in tqdm(image_files, desc='预测中'):
            # 预测
            prediction, probability = self.predict_image(str(image_path))
            
            # 保存掩码
            if save_mask:
                mask_path = mask_dir / f"{image_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), (prediction * 255).astype(np.uint8))
            
            # 保存可视化
            if visualize:
                # 读取原始图像
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 创建可视化（这里我们没有真实掩码，所以使用预测作为掩码）
                vis_path = vis_dir / f"{image_path.stem}_vis.png"
                
                # 简单的叠加可视化
                overlay = image.copy()
                mask_colored = np.zeros_like(image)
                mask_colored[prediction == 1] = [0, 255, 255]  # 青色表示水域
                
                result = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(vis_path), result)
        
        print(f"预测完成! 结果保存在: {output_dir}")
    
    def predict_with_ground_truth(self, image_path, mask_path, output_path=None):
        """
        预测并与真实标注对比
        
        Args:
            image_path: 图像路径
            mask_path: 真实掩码路径
            output_path: 输出路径（可选）
        """
        # 读取图像和掩码
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.max() > 1:
            mask = mask // 255  # 归一化到0-1
        
        # 预测
        prediction, _ = self.predict_image(image_path)
        
        # 可视化
        visualize_prediction(
            image=image,
            mask=mask,
            prediction=prediction,
            save_path=output_path
        )
        
        # 计算指标
        from src.utils.metrics import calculate_metrics
        
        metrics = calculate_metrics(
            torch.from_numpy(prediction),
            torch.from_numpy(mask),
            n_classes=self.config['model']['n_classes']
        )
        
        print("\n评估指标:")
        print(f"像素准确率: {metrics['pixel_accuracy']:.4f}")
        print(f"平均IoU: {metrics['mean_iou']:.4f}")
        print(f"平均Dice: {metrics['mean_dice']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        
        return prediction, metrics


def main():
    parser = argparse.ArgumentParser(description='潟湖水域分割预测')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录路径')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--visualize', action='store_true', help='保存可视化结果')
    parser.add_argument('--mask', type=str, help='真实掩码路径（用于对比）')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = Predictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # 预测
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图像预测
        if args.mask:
            # 与真实标注对比
            predictor.predict_with_ground_truth(
                image_path=str(input_path),
                mask_path=args.mask,
                output_path=str(Path(args.output) / 'comparison.png')
            )
        else:
            # 仅预测
            prediction, probability = predictor.predict_image(str(input_path))
            
            # 保存结果
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            mask_path = output_dir / f"{input_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), (prediction * 255).astype(np.uint8))
            
            print(f"预测完成! 掩码保存在: {mask_path}")
    
    elif input_path.is_dir():
        # 目录预测
        predictor.predict_directory(
            input_dir=str(input_path),
            output_dir=args.output,
            visualize=args.visualize,
            save_mask=True
        )
    
    else:
        print(f"错误: 输入路径不存在: {input_path}")


if __name__ == '__main__':
    main()

