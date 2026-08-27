<div align="center">

# ⚙️ Engine-RUL-Prediction

### Remaining useful life prediction for aero engines.

Machine-learning pipeline on the NASA C-MAPSS dataset — feature engineering, training and visualization.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NASA](https://img.shields.io/badge/Dataset-CMAPSS-blue)](https://www.nasa.gov/)

</div>

---

**Engine-RUL-Prediction** builds a machine-learning pipeline to predict the **remaining useful life (RUL)** of aero engines on the **NASA C-MAPSS** dataset — including feature engineering, model training, and visual analysis.

> [!NOTE]
> 中文项目：航空发动机寿命预测——NASA C-MAPSS 数据集，机器学习流水线，特征工程，可视化。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/Engine-RUL-Prediction.git
cd Engine-RUL-Prediction

pip install -r requirements.txt

# Train + predict
python src/main.py

# Visualization
python src/visualization.py
```

Data (`train_FD001.csv`, `test_FD001.csv`, `RUL_FD001.csv`) ships in `data/`.

---

## Features

- **C-MAPSS pipeline** — train/test/RUL on NASA FD001.
- **Feature engineering** — sensor feature construction and selection.
- **Visualization** — time series, correlation heatmap, prediction comparison.

---

## Project Structure

```
Engine-RUL-Prediction/
├── src/
│   ├── main.py            # pipeline entry
│   ├── rul_prediction.py  # model
│   └── visualization.py   # plots
├── data/                  # FD001 train/test/RUL CSVs
├── results/               # generated figures
└── docs/                  # design & usage docs
```

---

## License

MIT — free to use, modify and distribute.
