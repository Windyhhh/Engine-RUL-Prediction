"""
发动机剩余使用寿命（RUL）预测项目
使用神经网络算法对发动机数据进行处理、分析和RUL预测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RULPredictor:
    def __init__(self, train_file, test_file, rul_file):
        """初始化RUL预测器"""
        self.train_file = train_file
        self.test_file = test_file
        self.rul_file = rul_file
        self.train_data = None
        self.test_data = None
        self.rul_true = None
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.train_data = pd.read_csv(self.train_file, header=None)
        self.test_data = pd.read_csv(self.test_file, header=None)
        self.rul_true = pd.read_csv(self.rul_file, header=None).values.flatten()
        
        # 设置列名
        col_names = ['unit_id', 'time'] + [f'setting_{i}' for i in range(1, 4)] + \
                    [f'sensor_{i}' for i in range(1, 22)]
        self.train_data.columns = col_names
        self.test_data.columns = col_names
        
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.test_data.shape}")
        print(f"RUL真实值数量: {len(self.rul_true)}")
        
    def preprocess_data(self):
        """数据预处理"""
        print("\n正在进行数据预处理...")
        
        # 获取传感器列（从第6列开始）
        sensor_cols = [col for col in self.train_data.columns if col.startswith('sensor_')]
        
        # 标准化
        train_sensors = self.train_data[sensor_cols].values
        test_sensors = self.test_data[sensor_cols].values
        
        # 使用训练集的统计信息标准化
        train_sensors_scaled = self.scaler.fit_transform(train_sensors)
        test_sensors_scaled = self.scaler.transform(test_sensors)
        
        # 更新数据
        self.train_data[sensor_cols] = train_sensors_scaled
        self.test_data[sensor_cols] = test_sensors_scaled
        
        print("数据标准化完成")
        
    def create_sequences(self, data, unit_ids, sequence_length=30):
        """创建时间序列"""
        X, y, unit_list = [], [], []
        
        for unit_id in unit_ids:
            unit_data = data[data['unit_id'] == unit_id]
            sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
            values = unit_data[sensor_cols].values
            
            for i in range(len(values) - sequence_length):
                X.append(values[i:i+sequence_length])
                unit_list.append(unit_id)
            
            # 最后一个序列用于预测
            if len(values) >= sequence_length:
                X.append(values[-sequence_length:])
                unit_list.append(unit_id)
        
        return np.array(X), np.array(unit_list)
    
    def prepare_training_data(self, sequence_length=30):
        """准备训练数据"""
        print(f"\n正在准备训练数据（序列长度={sequence_length}）...")
        
        train_unit_ids = self.train_data['unit_id'].unique()
        X_train, train_units = self.create_sequences(self.train_data, train_unit_ids, sequence_length)
        
        # 创建RUL标签
        y_train = []
        for unit_id in train_units:
            unit_data = self.train_data[self.train_data['unit_id'] == unit_id]
            max_time = unit_data['time'].max()
            rul = max_time - unit_data['time'].max() + 1
            y_train.append(rul)
        
        y_train = np.array(y_train)
        
        print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
        return X_train, y_train
    
    def prepare_test_data(self, sequence_length=30):
        """准备测试数据"""
        print(f"\n正在准备测试数据...")
        
        test_unit_ids = self.test_data['unit_id'].unique()
        X_test_all = []
        test_unit_ids_list = []
        
        for unit_id in test_unit_ids:
            unit_data = self.test_data[self.test_data['unit_id'] == unit_id]
            sensor_cols = [col for col in self.test_data.columns if col.startswith('sensor_')]
            values = unit_data[sensor_cols].values
            
            if len(values) >= sequence_length:
                X_test_all.append(values[-sequence_length:])
                test_unit_ids_list.append(unit_id)
        
        X_test = np.array(X_test_all)
        print(f"测试数据形状: {X_test.shape}")
        return X_test, np.array(test_unit_ids_list)
    
    def build_model(self, input_shape):
        """构建LSTM模型"""
        print("\n正在构建LSTM模型...")
        
        model = Sequential([
            layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print(model.summary())
        return model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """训练模型"""
        print("\n正在训练模型...")
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def predict(self, X_test):
        """进行预测"""
        print("\n正在进行预测...")
        predictions = self.model.predict(X_test)
        return predictions.flatten()
    
    def evaluate_model(self, y_pred, y_true):
        """评估模型"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"\n模型评估结果:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return rmse, mae

if __name__ == "__main__":
    # 初始化预测器
    predictor = RULPredictor('../data/train_FD001.csv', '../data/test_FD001.csv', '../data/RUL_FD001.csv')
    
    # 加载数据
    predictor.load_data()
    
    # 数据预处理
    predictor.preprocess_data()
    
    # 准备训练数据
    X_train, y_train = predictor.prepare_training_data(sequence_length=30)
    
    # 准备测试数据
    X_test, test_unit_ids = predictor.prepare_test_data(sequence_length=30)
    
    # 训练模型
    history = predictor.train_model(X_train, y_train, epochs=50, batch_size=32)
    
    # 进行预测
    y_pred = predictor.predict(X_test)
    
    # 评估模型
    rmse, mae = predictor.evaluate_model(y_pred, predictor.rul_true[:len(y_pred)])
    
    print("\n程序执行完成！")

