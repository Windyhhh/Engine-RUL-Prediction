"""
发动机RUL预测主程序
整合数据处理、模型训练和可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("发动机剩余使用寿命（RUL）预测系统")
print("="*60)

# ============ 第一步：数据加载与预处理 ============
print("\n[步骤1] 数据加载与预处理")
print("-"*60)

# 加载数据（跳过第一行的中文列名）
try:
    train_data = pd.read_csv('../data/train_FD001.csv', header=None, skiprows=1, encoding='utf-8')
except:
    train_data = pd.read_csv('../data/train_FD001.csv', header=None, skiprows=1, encoding='latin-1')

try:
    test_data = pd.read_csv('../data/test_FD001.csv', header=None, skiprows=1, encoding='utf-8')
except:
    test_data = pd.read_csv('../data/test_FD001.csv', header=None, skiprows=1, encoding='latin-1')

try:
    rul_true = pd.read_csv('../data/RUL_FD001.csv', header=None, encoding='utf-8').values.flatten()
except:
    rul_true = pd.read_csv('../data/RUL_FD001.csv', header=None, encoding='latin-1').values.flatten()

# 设置列名
col_names = ['unit_id', 'time'] + [f'setting_{i}' for i in range(1, 4)] + \
            [f'sensor_{i}' for i in range(1, 22)]
train_data.columns = col_names
test_data.columns = col_names

print(f"✓ 训练集形状: {train_data.shape}")
print(f"✓ 测试集形状: {test_data.shape}")
print(f"✓ RUL真实值数量: {len(rul_true)}")
print(f"✓ 发动机总数: {train_data['unit_id'].max()}")

# 数据标准化
sensor_cols = [col for col in train_data.columns if col.startswith('sensor_')]
scaler = StandardScaler()

train_sensors = train_data[sensor_cols].values
test_sensors = test_data[sensor_cols].values

train_sensors_scaled = scaler.fit_transform(train_sensors)
test_sensors_scaled = scaler.transform(test_sensors)

train_data[sensor_cols] = train_sensors_scaled
test_data[sensor_cols] = test_sensors_scaled

print("✓ 数据标准化完成")

# ============ 第二步：数据可视化 ============
print("\n[步骤2] 数据可视化")
print("-"*60)

# 2.1 时间序列趋势图
print("生成时间序列趋势图...")
selected_units = [1, 2, 3]
selected_sensors = sensor_cols[:6]

fig, axes = plt.subplots(len(selected_sensors), 1, figsize=(14, 12))
fig.suptitle('发动机传感器数据时间序列趋势（训练集）', fontsize=16, fontweight='bold')

for idx, sensor in enumerate(selected_sensors):
    ax = axes[idx]
    for unit_id in selected_units:
        unit_data = train_data[train_data['unit_id'] == unit_id]
        ax.plot(unit_data['time'], unit_data[sensor], label=f'发动机 {unit_id}', linewidth=1.5)
    
    ax.set_ylabel(sensor, fontsize=10)
    ax.set_xlabel('时间周期' if idx == len(selected_sensors)-1 else '')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/01_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 时间序列趋势图已保存: 01_time_series.png")

# 2.2 相关性热力图
print("生成相关性热力图...")
correlation_matrix = train_data[sensor_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('传感器数据相关性热力图', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../results/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 相关性热力图已保存: 02_correlation_heatmap.png")

# 2.3 分布图
print("生成特征分布图...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('发动机传感器数据分布（训练集）', fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, sensor in enumerate(selected_sensors):
    ax = axes[idx]
    for unit_id in selected_units:
        unit_data = train_data[train_data['unit_id'] == unit_id]
        ax.hist(unit_data[sensor], bins=30, alpha=0.5, label=f'发动机 {unit_id}')
    
    ax.set_xlabel(sensor, fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/03_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 特征分布图已保存: 03_distribution.png")

# ============ 第三步：创建时间序列数据 ============
print("\n[步骤3] 创建时间序列数据")
print("-"*60)

def create_sequences(data, unit_ids, sequence_length=30):
    """创建时间序列和对应的RUL标签"""
    X, y, unit_list = [], [], []

    for unit_id in unit_ids:
        unit_data = data[data['unit_id'] == unit_id].sort_values('time')
        values = unit_data[sensor_cols].values
        times = unit_data['time'].values
        max_time = times[-1]

        # 为每个时间窗口创建序列
        for i in range(len(values) - sequence_length + 1):
            X.append(values[i:i+sequence_length])
            # RUL = 该发动机的最大时间 - 当前窗口结束时的时间 + 1
            current_time = times[i + sequence_length - 1]
            rul = max_time - current_time + 1
            y.append(rul)
            unit_list.append(unit_id)

    return np.array(X), np.array(y), np.array(unit_list)

sequence_length = 30
train_unit_ids = train_data['unit_id'].unique()
X_train, y_train, train_units = create_sequences(train_data, train_unit_ids, sequence_length)

print(f"✓ 训练数据形状: X={X_train.shape}, y={y_train.shape}")

# 准备测试数据
test_unit_ids_unique = sorted(test_data['unit_id'].unique())
X_test_all = []
y_test_all = []
test_unit_ids_list = []

for unit_id in test_unit_ids_unique:
    unit_data = test_data[test_data['unit_id'] == unit_id]
    values = unit_data[sensor_cols].values

    if len(values) >= sequence_length:
        X_test_all.append(values[-sequence_length:])
        test_unit_ids_list.append(unit_id)
        # 测试集的RUL来自rul_true
        y_test_all.append(rul_true[unit_id - 1])

X_test = np.array(X_test_all)
y_test = np.array(y_test_all)
test_unit_ids_array = np.array(test_unit_ids_list)

print(f"✓ 测试数据形状: {X_test.shape}")
print(f"✓ 测试标签形状: {y_test.shape}")

# ============ 第四步：构建和训练模型 ============
print("\n[步骤4] 构建和训练LSTM模型")
print("-"*60)

model = Sequential([
    layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("✓ 模型构建完成")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("开始训练模型...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
print("✓ 模型训练完成")

# ============ 第五步：模型预测与评估 ============
print("\n[步骤5] 模型预测与评估")
print("-"*60)

y_pred = model.predict(X_test, verbose=0).flatten()

# 计算误差指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"✓ RMSE: {rmse:.4f}")
print(f"✓ MAE: {mae:.4f}")

# ============ 第六步：预测结果可视化 ============
print("\n[步骤6] 预测结果可视化")
print("-"*60)

# 6.1 选定发动机的预测对比图（柱状图对比）
print("生成选定发动机的预测对比图...")
fig, ax = plt.subplots(figsize=(12, 6))

# 收集3台发动机的预测值和真实值
unit_ids_list = []
pred_values = []
true_values = []

for unit_id in selected_units:
    unit_mask = test_unit_ids_array == unit_id
    unit_indices = np.where(unit_mask)[0]

    if len(unit_indices) > 0:
        unit_ids_list.append(f'发动机{unit_id}')
        pred_values.append(y_pred[unit_indices][0])
        true_values.append(y_test[unit_indices][0])

# 绘制柱状图对比
x = np.arange(len(unit_ids_list))
width = 0.35

bars1 = ax.bar(x - width/2, true_values, width, label='真实RUL值', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, pred_values, width, label='预测RUL值', color='#A23B72', alpha=0.8)

# 在柱子上添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('RUL (周期)', fontsize=12, fontweight='bold')
ax.set_xlabel('发动机编号', fontsize=12, fontweight='bold')
ax.set_title('选定发动机的RUL预测值与真实值对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(unit_ids_list, fontsize=11)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../results/04_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 预测对比图已保存: 04_prediction_comparison.png")

# 6.2 所有预测结果
print("生成所有预测结果图...")
fig, ax = plt.subplots(figsize=(14, 6))

x_pos = np.arange(len(y_pred))
ax.plot(x_pos, y_test, 'o-', label='真实值', linewidth=1.5, markersize=4)
ax.plot(x_pos, y_pred, 's--', label='预测值', linewidth=1.5, markersize=4, alpha=0.7)

ax.set_xlabel('测试样本序号', fontsize=12)
ax.set_ylabel('RUL (周期)', fontsize=12)
ax.set_title('所有测试样本的RUL预测结果', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/05_all_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 所有预测结果图已保存: 05_all_predictions.png")

# ============ 完成 ============
print("\n" + "="*60)
print("✓ 程序执行完成！")
print("="*60)
print("\n生成的文件:")
print("  - 01_time_series.png: 时间序列趋势图")
print("  - 02_correlation_heatmap.png: 相关性热力图")
print("  - 03_distribution.png: 特征分布图")
print("  - 04_prediction_comparison.png: 预测对比图")
print("  - 05_all_predictions.png: 所有预测结果图")
print("\n模型性能:")
print(f"  - RMSE: {rmse:.4f}")
print(f"  - MAE: {mae:.4f}")
print("="*60)

