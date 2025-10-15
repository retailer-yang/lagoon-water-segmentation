# 使用指南

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/your-username/lagoon-water-segmentation.git
cd lagoon-water-segmentation

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将你的数据组织成以下结构：

```
data/
├── processed/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
```

- `images/`: 存放原始图像（支持 .jpg, .png, .tif 等格式）
- `masks/`: 存放对应的分割掩码（二值图像，0=背景，1=水域）

### 3. 训练模型

#### 使用U-Net模型

```bash
python scripts/train.py --config configs/unet_config.yaml
```

#### 使用DeepLabV3+模型

```bash
python scripts/train.py --config configs/deeplabv3_config.yaml
```

#### 恢复训练

```bash
python scripts/train.py --config configs/unet_config.yaml --resume results/unet_experiment/checkpoint_latest.pth
```

### 4. 预测

#### 预测单张图像

```bash
python scripts/predict.py \
    --model results/unet_experiment/checkpoint_best.pth \
    --config configs/unet_config.yaml \
    --input test_image.jpg \
    --output results/predictions \
    --visualize
```

#### 预测整个目录

```bash
python scripts/predict.py \
    --model results/unet_experiment/checkpoint_best.pth \
    --config configs/unet_config.yaml \
    --input data/test/images \
    --output results/predictions \
    --visualize
```

#### 与真实标注对比

```bash
python scripts/predict.py \
    --model results/unet_experiment/checkpoint_best.pth \
    --config configs/unet_config.yaml \
    --input test_image.jpg \
    --output results/predictions \
    --mask test_mask.png
```

### 5. 评估模型

```bash
python scripts/evaluate.py \
    --model results/unet_experiment/checkpoint_best.pth \
    --config configs/unet_config.yaml \
    --data_dir data/processed/val/images \
    --mask_dir data/processed/val/masks \
    --output results/evaluation_results.txt
```

## 配置文件说明

配置文件位于 `configs/` 目录，使用YAML格式。主要参数：

### 模型配置

```yaml
model:
  name: unet  # 模型名称: unet 或 deeplabv3
  n_classes: 2  # 分类类别数
  params:
    n_channels: 3  # 输入通道数
    bilinear: true  # 是否使用双线性插值
```

### 数据配置

```yaml
data:
  train_images: data/processed/train/images
  train_masks: data/processed/train/masks
  val_images: data/processed/val/images
  val_masks: data/processed/val/masks
  image_size: 512  # 训练图像尺寸
```

### 训练配置

```yaml
training:
  batch_size: 8
  num_workers: 4
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: Adam  # Adam, SGD, AdamW
  scheduler: ReduceLROnPlateau
  criterion: CombinedLoss  # CrossEntropyLoss, DiceLoss, CombinedLoss
```

## Jupyter Notebook示例

查看 `notebooks/demo.ipynb` 获取交互式使用示例。

启动Jupyter Notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

## 项目结构

```
lagoon-water-segmentation/
├── configs/               # 配置文件
│   ├── unet_config.yaml
│   └── deeplabv3_config.yaml
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── models/               # 保存的模型
├── notebooks/            # Jupyter笔记本
│   └── demo.ipynb
├── results/              # 结果输出
├── scripts/              # 训练和推理脚本
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── src/                  # 源代码
│   ├── data/            # 数据处理
│   ├── models/          # 模型定义
│   └── utils/           # 工具函数
├── requirements.txt      # 依赖包
└── README.md            # 项目说明
```

## 常见问题

### 1. CUDA out of memory

减小batch_size或image_size：

```yaml
training:
  batch_size: 4  # 从8减到4
data:
  image_size: 256  # 从512减到256
```

### 2. 数据加载错误

确保图像和掩码文件名匹配：
- `image1.jpg` 对应 `image1.png`
- 掩码必须是单通道灰度图

### 3. 训练不收敛

尝试：
- 调整学习率
- 更换损失函数
- 增加数据增强
- 使用预训练模型（DeepLabV3+）

## 技术支持

如有问题，请通过GitHub Issues反馈。

