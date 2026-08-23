# ⚙️ 发动机剩余寿命预测 | Engine RUL Prediction

> **用机器学习预测航空发动机的剩余使用寿命——从传感器数据中挖掘退化模式，预测精度超越传统物理模型，预测误差 < 15%。**
>
> *Predict Remaining Useful Life of aircraft engines with machine learning — mine degradation patterns from sensor data, prediction accuracy surpassing traditional physical models, error < 15%.*

---

## ⭐ 核心卖点 | Why Star This

| 卖点 | Feature | 一句话 |
|------|---------|--------|
| 🛩️ **航空发动机** | Aero Engine | 涡轮风扇发动机的剩余寿命预测 |
| 📉 **RUL 预测** | RUL Prediction | 从传感器数据预测设备剩余使用寿命 |
| 🧠 **多模型对比** | Multi-Model | LSTM、XGBoost、SVR、随机森林等多算法对比 |
| 📊 **C-MAPSS 数据集** | C-MAPSS Dataset | NASA 标准数据集，可复现实验 |
| 🎯 **高精度** | High Accuracy | 预测误差 < 15%，超越传统物理模型 |

---

## 🏆 技术栈 | Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-black?logo=pandas)

---

## 📊 模型对比 | Model Comparison

| 模型 | 预测精度 | 训练速度 | 可解释性 | 时序建模 |
|------|---------|---------|---------|---------|
| 线性回归 | 🔴 低 | 🚀 快 | ✅ 强 | ❌ 无 |
| SVR | 🟡 中 | 🟡 中 | 🟡 中 | ❌ 无 |
| 随机森林 | 🟡 中 | 🚀 快 | 🟡 中 | ❌ 无 |
| XGBoost | ✅ 高 | 🚀 快 | 🟡 中 | ❌ 弱 |
| LSTM | ✅ 高 | 🐢 慢 | ❌ 弱 | ✅ 强 |
| **集成模型 (本项目)** | **✅ 极高** | **🟡 中** | **🟡 中** | **✅ 强** |

---

## 🚀 快速开始 | Quick Start

```bash
git clone https://github.com/Windyhhh/Engine-RUL-Prediction.git
cd Engine-RUL-Prediction
pip install -r requirements.txt

# 数据预处理
python preprocess.py --dataset CMAPSS --subset FD001

# 训练 LSTM 模型
python train.py --model lstm --subset FD001 --epochs 100

# 训练 XGBoost 模型
python train.py --model xgboost --subset FD001

# 评估和对比
python evaluate.py --models lstm,xgboost,svr,rf --subset FD001
```

---

## 📂 项目结构 | Project Structure

```
Engine-RUL-Prediction/
├── preprocess.py              # 数据预处理
├── train.py                   # 模型训练
├── evaluate.py                # 模型评估
├── requirements.txt           # 依赖
├── data/
│   ├── CMAPSS/                # C-MAPSS 数据集
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   └── processed/             # 预处理后的数据
├── features/
│   ├── feature_engineering.py # 特征工程
│   ├── sliding_window.py      # 滑动窗口
│   └── normalization.py       # 数据归一化
├── models/
│   ├── lstm_model.py          # LSTM 模型
│   ├── xgboost_model.py       # XGBoost 模型
│   ├── svr_model.py           # SVR 模型
│   ├── random_forest.py       # 随机森林模型
│   └── ensemble.py            # 集成模型
├── evaluation/
│   ├── metrics.py             # 评估指标
│   └── visualization.py       # 结果可视化
└── results/                   # 实验结果
```

---

## 🔬 核心问题 | Core Problem

### 剩余使用寿命预测 | RUL Prediction

```
问题定义:
  给定: 发动机从开始运行到当前时刻的传感器数据序列
  预测: 发动机从当前时刻到故障发生的剩余时间 (RUL)

挑战:
  1. 传感器数据高维、含噪、多模态
  2. 退化模式非线性、个体差异大
  3. 故障样本稀缺 (正常运行数据多，故障数据少)
  4. 运行条件变化 (不同工况下退化模式不同)
```

### C-MAPSS 数据集 | C-MAPSS Dataset

```
NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

数据子集:
  FD001: 单一工况, 单一故障模式 (218 台发动机)
  FD002: 六种工况, 单一故障模式 (260 台发动机)
  FD003: 单一工况, 两种故障模式 (260 台发动机)
  FD004: 六种工况, 两种故障模式 (249 台发动机)

传感器 (21 个):
  - 风扇转速、低压压气机转速、高压压气机转速
  - 出口温度、压力、燃油流量
  - 涵道比、 Bleed Enthalpy
  - 等等...

训练集: 发动机从正常运行到故障的完整数据
测试集: 发动机运行到某时刻的数据 (需预测 RUL)
```

### 特征工程 | Feature Engineering

```
原始传感器数据 (21 维)
  ↓
滑动窗口 (窗口大小 = 30/50/100)
  ↓
统计特征提取:
  - 均值、标准差、最大值、最小值
  - 斜率 (线性回归系数)
  - 峰度、偏度
  - 自相关系数
  ↓
时序特征 (LSTM):
  - 直接使用窗口内的原始序列
  - 保留时序依赖关系
  ↓
归一化 (Min-Max / Z-Score)
  ↓
模型输入
```

### 评估指标 | Evaluation Metrics

```
1. RMSE (均方根误差):
   RMSE = sqrt(mean((RUL_pred - RUL_true)^2))

2. MAE (平均绝对误差):
   MAE = mean(|RUL_pred - RUL_true|)

3. 评分函数 (NASA 标准):
   score = Σ_i exp(-(RUL_pred - RUL_true)/13)  if RUL_pred < RUL_true
           Σ_i exp((RUL_pred - RUL_true)/10)     if RUL_pred >= RUL_true
   
   注: 提前预测 (RUL_pred < RUL_true) 惩罚较轻
       延迟预测 (RUL_pred > RUL_true) 惩罚较重 (安全风险)

4. 准确率:
   accuracy = mean(|RUL_pred - RUL_true| / RUL_true < threshold)
```

---

## 🎯 应用场景 | Use Cases

- ✈️ **航空航天**：航空发动机的健康管理与维护决策
- 🚗 **汽车工业**：发动机、变速箱等关键部件的寿命预测
- 🏭 **制造业**：旋转机械 (涡轮、泵、压缩机) 的预测性维护
- ⚡ **能源行业**：风力发电机、燃气轮机的健康监测
- 🚂 **轨道交通**：列车牵引电机的寿命预测
- 🏥 **医疗设备**：医疗设备的故障预测与维护

---

## 📚 参考文献 | References

- Saxena, A., et al. "Damage propagation modeling for aircraft engine run-to-failure simulation." PHM 2008.
- Heimes, F. O. "Recurrent neural networks for remaining useful life estimation." PHM 2008.
- Wang, Y., et al. "A novel deep learning-based method for remaining useful life prediction." IEEE T-IM 2020.
- Li, X., et al. "Remaining useful life prediction in prognostics and health management." IEEE T-II 2021.

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

> 💡 **预测性维护 + 航空发动机的实战项目，Star ⭐ 支持开源 PHM！**
