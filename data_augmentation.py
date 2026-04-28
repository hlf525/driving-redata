"""
数据增强脚本 - 用于分心驾驶数据集
支持光照调整、噪音添加、遮挡等多种增强方式
"""
import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


class DataAugmentor:
    """数据增强器类"""
    
    def __init__(self, config):
        """
        初始化数据增强器
        config: 配置字典，包含各种增强参数
        """
        self.config = config
        
    def adjust_brightness(self, image, factor):
        """
        调整图像亮度
        factor: 亮度因子，<1变暗，>1变亮，范围建议[0.5, 1.5]
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_gamma(self, image, gamma):
        """
        Gamma 校正
        gamma < 1: 图像变亮（暗部增强）
        gamma > 1: 图像变暗（亮部压缩）
        推荐范围: [0.6, 1.6]
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        return cv2.LUT(image, table)

    def adjust_contrast(self, image, factor):
        """
        调整图像对比度
        factor: 对比度因子，<1降低对比度，>1增加对比度，范围建议[0.5, 1.5]
        """
        mean = np.mean(image)
        result = (image - mean) * factor + mean
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_gaussian_noise(self, image, mean=0, std=25):
        """
        添加高斯噪声
        mean: 噪声均值
        std: 噪声标准差，范围建议[10, 50]
        """
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        """
        添加椒盐噪声
        salt_prob: 盐噪声概率，范围建议[0.001, 0.05]
        pepper_prob: 椒噪声概率，范围建议[0.001, 0.05]
        """
        noisy_image = image.copy()
        
        # 盐噪声（白点）
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        noisy_image[salt_mask] = 255
        
        # 椒噪声（黑点）
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def add_random_occlusion(self, image, num_blocks=1, block_size_ratio=(0.1, 0.3)):
        """
        添加随机遮挡
        num_blocks: 遮挡块的数量
        block_size_ratio: 遮挡块大小占图像的比例范围 (min_ratio, max_ratio)
        """
        occluded_image = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_blocks):
            # 随机遮挡块大小
            block_h = int(h * np.random.uniform(*block_size_ratio))
            block_w = int(w * np.random.uniform(*block_size_ratio))
            
            # 随机遮挡位置
            x = np.random.randint(0, max(1, w - block_w))
            y = np.random.randint(0, max(1, h - block_h))
            
            # 随机遮挡颜色（黑色或随机颜色）
            if np.random.random() < 0.5:
                color = (0, 0, 0)  # 黑色
            else:
                color = tuple(np.random.randint(0, 256, 3).tolist())
            
            cv2.rectangle(occluded_image, (x, y), (x + block_w, y + block_h), color, -1)
        
        return occluded_image
    
    def add_motion_blur(self, image, kernel_size=15):
        """
        添加运动模糊
        kernel_size: 模糊核大小，范围建议[5, 25]
        """
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)
    
    def adjust_saturation(self, image, factor):
        """
        调整饱和度
        factor: 饱和度因子，<1降低饱和度，>1增加饱和度，范围建议[0.5, 1.5]
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def augment_image(self, image, aug_params):
        """
        对单张图像进行增强
        aug_params: 增强参数字典
        """
        result = image.copy()
        
        # 光照调整
        if aug_params.get('brightness'):
            result = self.adjust_brightness(result, aug_params['brightness'])
        
        if aug_params.get('contrast'):
            result = self.adjust_contrast(result, aug_params['contrast'])
        
        if aug_params.get('saturation'):
            result = self.adjust_saturation(result, aug_params['saturation'])
        
        # Gamma 校正（显示模拟）
        if aug_params.get('gamma'):
            result = self.adjust_gamma(result, aug_params['gamma'])

        # 噪声添加
        if aug_params.get('gaussian_noise'):
            result = self.add_gaussian_noise(result, std=aug_params['gaussian_noise'])
        
        if aug_params.get('salt_pepper_noise'):
            salt_prob, pepper_prob = aug_params['salt_pepper_noise']
            result = self.add_salt_pepper_noise(result, salt_prob, pepper_prob)
        
        # 遮挡
        if aug_params.get('occlusion'):
            num_blocks, block_size_ratio = aug_params['occlusion']
            result = self.add_random_occlusion(result, num_blocks, block_size_ratio)
        
        # 运动模糊
        if aug_params.get('motion_blur'):
            result = self.add_motion_blur(result, aug_params['motion_blur'])
        
        return result
    
    def process_dataset(self, input_dir, output_dir, augmentation_configs):
        """
        处理整个数据集
        input_dir: 输入数据集目录
        output_dir: 输出数据集目录
        augmentation_configs: 增强配置列表，每个配置对应一种增强方式
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 检查输入目录是否直接包含图像文件
        direct_images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        
        if direct_images:
            # 输入目录直接包含图像文件，直接处理
            print(f"\n处理目录: {input_path.name}")
            self._process_class_directory(input_path, output_path, augmentation_configs)
        else:
            # 输入目录包含子目录，遍历处理每个子目录
            for class_dir in input_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                print(f"\n处理类别: {class_name}")
                
                # 为每个类别创建输出目录
                output_class_dir = output_path / class_name
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                self._process_class_directory(class_dir, output_class_dir, augmentation_configs)
    
    def _process_class_directory(self, class_dir, output_dir, augmentation_configs):
        """
        处理单个类别目录
        """
        # 获取该类别下的所有图像
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        if not image_files:
            print(f"警告: {class_dir} 下没有找到图像文件")
            return
        
        # 处理每张图像
        for img_file in tqdm(image_files, desc=f"处理 {class_dir.name}"):
            # 读取原始图像
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"无法读取图像: {img_file}")
                continue
            
            # 保存原始图像（如果配置要求）
            if self.config.get('keep_original', True):
                output_file = output_dir / img_file.name
                cv2.imwrite(str(output_file), image)
            
            # 应用每种增强配置
            for idx, aug_config in enumerate(augmentation_configs):
                augmented_image = self.augment_image(image, aug_config)
                
                # 生成新的文件名
                stem = img_file.stem
                suffix = img_file.suffix
                aug_name = aug_config.get('name', f'aug{idx}')
                new_filename = f"{stem}_{aug_name}{suffix}"
                
                output_file = output_dir / new_filename
                cv2.imwrite(str(output_file), augmented_image)


def load_config(config_file):
    """从JSON文件加载配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='数据增强脚本')
    parser.add_argument('--input', '-i', required=True, help='输入数据集目录')
    parser.add_argument('--output', '-o', required=True, help='输出数据集目录')
    parser.add_argument('--config', '-c', required=True, help='配置文件路径(JSON)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建增强器
    augmentor = DataAugmentor(config)
    
    # 处理数据集
    augmentor.process_dataset(
        args.input,
        args.output,
        config.get('augmentations', [])
    )
    
    print("\n数据增强完成！")


if __name__ == '__main__':
    main()
