"""
数据可视化模块
包括时间序列趋势图、相关性热力图、分布图和预测结果对比图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)

class DataVisualizer:
    def __init__(self, train_data, test_data, rul_true):
        """初始化可视化器"""
        self.train_data = train_data
        self.test_data = test_data
        self.rul_true = rul_true
        
    def plot_time_series(self, selected_units=[1, 2, 3], save_path='time_series.png'):
        """绘制时间序列趋势图"""
        print("正在绘制时间序列趋势图...")
        
        sensor_cols = [col for col in self.train_data.columns if col.startswith('sensor_')]
        selected_sensors = sensor_cols[:6]  # 选择前6个传感器
        
        fig, axes = plt.subplots(len(selected_sensors), 1, figsize=(14, 12))
        fig.suptitle('发动机传感器数据时间序列趋势（训练集）', fontsize=16, fontweight='bold')
        
        for idx, sensor in enumerate(selected_sensors):
            ax = axes[idx]
            for unit_id in selected_units:
                unit_data = self.train_data[self.train_data['unit_id'] == unit_id]
                ax.plot(unit_data['time'], unit_data[sensor], label=f'发动机 {unit_id}', linewidth=1.5)
            
            ax.set_ylabel(sensor, fontsize=10)
            ax.set_xlabel('时间周期' if idx == len(selected_sensors)-1 else '')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"时间序列趋势图已保存: {save_path}")
        plt.close()
    
    def plot_correlation_heatmap(self, save_path='correlation_heatmap.png'):
        """绘制相关性热力图"""
        print("正在绘制相关性热力图...")
        
        sensor_cols = [col for col in self.train_data.columns if col.startswith('sensor_')]
        correlation_matrix = self.train_data[sensor_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('传感器数据相关性热力图', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相关性热力图已保存: {save_path}")
        plt.close()
    
    def plot_distribution(self, selected_units=[1, 2, 3], save_path='distribution.png'):
        """绘制特征分布图"""
        print("正在绘制特征分布图...")
        
        sensor_cols = [col for col in self.train_data.columns if col.startswith('sensor_')]
        selected_sensors = sensor_cols[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('发动机传感器数据分布（训练集）', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, sensor in enumerate(selected_sensors):
            ax = axes[idx]
            for unit_id in selected_units:
                unit_data = self.train_data[self.train_data['unit_id'] == unit_id]
                ax.hist(unit_data[sensor], bins=30, alpha=0.5, label=f'发动机 {unit_id}')
            
            ax.set_xlabel(sensor, fontsize=10)
            ax.set_ylabel('频数', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存: {save_path}")
        plt.close()
    
    def plot_prediction_comparison(self, y_pred, test_unit_ids, selected_units=[1, 2, 3],
                                   save_path='prediction_comparison.png'):
        """绘制预测值与真实值对比图"""
        print("正在绘制预测值与真实值对比图...")
        
        fig, axes = plt.subplots(len(selected_units), 1, figsize=(14, 10))
        fig.suptitle('RUL预测值与真实值对比', fontsize=16, fontweight='bold')
        
        if len(selected_units) == 1:
            axes = [axes]
        
        for idx, unit_id in enumerate(selected_units):
            ax = axes[idx]
            
            # 找到该发动机的预测结果
            unit_mask = test_unit_ids == unit_id
            unit_indices = np.where(unit_mask)[0]
            
            if len(unit_indices) > 0:
                unit_pred = y_pred[unit_indices]
                unit_true = self.rul_true[unit_indices]
                
                x_pos = np.arange(len(unit_pred))
                ax.plot(x_pos, unit_true, 'o-', label='真实值', linewidth=2, markersize=6)
                ax.plot(x_pos, unit_pred, 's--', label='预测值', linewidth=2, markersize=6)
                
                ax.set_ylabel('RUL (周期)', fontsize=11)
                ax.set_xlabel('样本序号' if idx == len(selected_units)-1 else '')
                ax.set_title(f'发动机 {unit_id}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测对比图已保存: {save_path}")
        plt.close()
    
    def plot_all_predictions(self, y_pred, save_path='all_predictions.png'):
        """绘制所有预测结果"""
        print("正在绘制所有预测结果...")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x_pos = np.arange(len(y_pred))
        ax.plot(x_pos, self.rul_true[:len(y_pred)], 'o-', label='真实值', linewidth=1.5, markersize=4)
        ax.plot(x_pos, y_pred, 's--', label='预测值', linewidth=1.5, markersize=4, alpha=0.7)
        
        ax.set_xlabel('测试样本序号', fontsize=12)
        ax.set_ylabel('RUL (周期)', fontsize=12)
        ax.set_title('所有测试样本的RUL预测结果', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"所有预测结果图已保存: {save_path}")
        plt.close()

def visualize_all(train_data, test_data, rul_true, y_pred, test_unit_ids):
    """执行所有可视化"""
    visualizer = DataVisualizer(train_data, test_data, rul_true)
    
    # 绘制时间序列
    visualizer.plot_time_series(selected_units=[1, 2, 3])
    
    # 绘制相关性热力图
    visualizer.plot_correlation_heatmap()
    
    # 绘制分布图
    visualizer.plot_distribution(selected_units=[1, 2, 3])
    
    # 绘制预测对比图
    visualizer.plot_prediction_comparison(y_pred, test_unit_ids, selected_units=[1, 2, 3])
    
    # 绘制所有预测结果
    visualizer.plot_all_predictions(y_pred)
    
    print("\n所有可视化图表已生成！")

