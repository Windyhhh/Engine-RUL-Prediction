# 🛩️ Engine RUL Prediction | 发动机剩余寿命预测系统

> **Remaining Useful Life (RUL) prediction for aircraft turbofan engines using NASA C-MAPSS dataset. Machine learning pipeline with feature engineering, model training, and comprehensive visualization.**
>
> 基于 NASA C-MAPSS 数据集的航空涡扇发动机剩余寿命（RUL）预测。机器学习流水线，含特征工程、模型训练和全面可视化。

---

## 🌟 Features | 核心特性

- **NASA C-MAPSS Dataset** — FD001 subset (100 training engines, 100 test engines)
- **Feature Engineering** — 21 sensor measurements + 3 operational settings
- **RUL Labeling** — Piecewise linear degradation model
- **ML Pipeline** — Data preprocessing → feature extraction → model training → evaluation
- **Visualization** — Time series, correlation heatmap, distribution, prediction comparison
- **Modular Design** — Separate modules for prediction, visualization, and main entry

---

## 📁 Project Structure | 项目结构

```
Engine-RUL-Prediction/
├── src/
│   ├── main.py                # Main entry point
│   ├── rul_prediction.py      # Core RUL prediction logic
│   └── visualization.py       # Visualization utilities
├── data/
│   ├── train_FD001.csv       # Training data (NASA C-MAPSS FD001)
│   ├── test_FD001.csv        # Test data
│   └── RUL_FD001.csv         # Ground truth RUL for test set
├── docs/
│   ├── 使用说明.md
│   ├── 功能说明.md
│   ├── 设计思路.md
│   ├── 结果说明.md
│   └── 爆款博客.md
├── results/
│   ├── 01_time_series.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_distribution.png
│   ├── 04_prediction_comparison.png
│   └── 05_all_predictions.png
└── README.md
```

---

## 🚀 Quick Start | 快速开始

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

# Run full pipeline
python src/main.py

# Run prediction only
python src/rul_prediction.py

# Generate visualizations
python src/visualization.py
```

---

## 📊 Dataset | 数据集

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**

| Property | Value |
|----------|-------|
| Training engines | 100 |
| Test engines | 100 |
| Sensors | 21 |
| Operational settings | 3 |
| Max cycles (train) | ~362 |
| Max cycles (test) | ~341 |

Each row contains: engine_id, cycle, setting_1-3, sensor_1-21.

---

## 🔬 Methodology | 方法

1. **Data Preprocessing** — Clean, normalize, handle missing values
2. **Feature Engineering** — Rolling statistics, trend features, sensor selection
3. **RUL Labeling** — Piecewise linear: RUL = min(actual_cycles_remaining, max_rul_threshold)
4. **Model Training** — Regression model (Random Forest / XGBoost / LSTM)
5. **Evaluation** — RMSE, MAE, score function (asymmetric penalty)

---

## 📈 Evaluation Metrics | 评估指标

- **RMSE** — Root Mean Square Error
- **MAE** — Mean Absolute Error
- **Score** — NASA's asymmetric scoring function (heavier penalty for late predictions)

---

## 📄 License | 许可证

MIT License.

---

<div align="center">

**Built with 🛩️ for predictive maintenance research**

[GitHub](https://github.com/Windyhhh/Engine-RUL-Prediction)

</div>
