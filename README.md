# Diabetes Digital Twin

A machine learning and physiological modeling framework for Continuous Glucose Monitoring (CGM) analytics, glucose spike prediction, and clinical context synthesis using the ShanghaiT2DM dataset.

---

## Overview

The **Diabetes Digital Twin** project models personalized blood glucose dynamics to predict impending glucose spikes by combining:
- **CGM Trajectories:** High-frequency interstitial glucose measurements.
- **Contextual Signals:** Meal ingestion events and insulin administration over 30-minute, 60-minute, and 120-minute rolling windows.
- **Clinical EHR Metadata:** Patient demographics, baseline metabolic indicators, and medication history.
- **Patient-Level Grouping:** Leakage-free train/validation/test splits strictly separated by patient ID.

---

## Repository Structure

```text
diabetes-digital-twin/
├── data/
│   ├── raw/                  # Raw ShanghaiT2DM dataset archives (.zip / .xls / .xlsx)
│   └── processed/            # Processed features and benchmark metrics
│       ├── model_comparison.csv       # Benchmark metrics across trained models
│       └── rf_threshold_results.csv   # Threshold optimization sweep results
├── ml/                       # Machine Learning & Feature Engineering Pipeline
│   ├── data_loader.py        # Ingests raw Excel CGM sessions and joins patient EHR summary
│   ├── diagnose_data.py      # Session and data integrity diagnostics
│   ├── make_labels.py        # Computes glucose spike target labels
│   ├── analyze_spikes.py     # Class balance and spike frequency analysis
│   ├── sanity_check.py       # Data sanity and continuity verification
│   ├── analyze_context.py    # Exploratory analysis of meal and insulin signals
│   ├── analyze_context_signal.py # Statistical signal strength evaluation of context events
│   ├── build_features.py     # Baseline feature engineering (V1)
│   ├── build_features_v2.py  # Timestamp-aligned rolling context features (V2)
│   ├── split_data.py         # Patient-grouped train/val/test dataset split
│   ├── train_models.py       # Model training (Random Forest, XGBoost, Logistic Regression)
│   └── tune_threshold.py     # Decision threshold optimization for recall/precision trade-off
├── digital_twin/             # Physiological simulation & state estimation modules
├── mcp_server/               # Model Context Protocol (MCP) server integration
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- Virtual environment tool (`venv`)

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/kartiksrathod/diabetes-digital-twin.git
cd diabetes-digital-twin

python -m venv .venv

# On Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# On Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Machine Learning Pipeline

### 1. Data Ingestion & Harmonization
Ingest raw ShanghaiT2DM CGM files and align them with patient EHR data:
```bash
python ml/data_loader.py --zip "data/raw/diabetes_datasets.zip"
```
*Output: `data/processed/shanghai_t2dm_master.csv`*

### 2. Label Generation
Generate target labels (`spike_label`) based on glucose thresholds and delta windows:
```bash
python ml/make_labels.py
```
*Output: `data/processed/shanghai_t2dm_labeled.csv`*

### 3. Context Feature Engineering (V2)
Extract timestamp-aligned rolling context features (meal events, insulin doses, medication history, and glucose lag dynamics):
```bash
python ml/build_features_v2.py
```
*Output: `data/processed/shanghai_t2dm_features_v2.csv`*

### 4. Patient-Grouped Split
Split the dataset into Train (70%), Validation (15%), and Test (15%) partitions grouped strictly by `patient_id` to prevent data leakage:
```bash
python ml/split_data.py
```
*Outputs: `train.csv`, `validation.csv`, `test.csv` in `data/processed/`*

### 5. Model Training & Evaluation
Train and evaluate baseline classification models on validation data:
```bash
python ml/train_models.py
```
*Output: `data/processed/model_comparison.csv`*

### 6. Threshold Optimization
Tune decision probability thresholds on the best-performing model to balance clinical sensitivity (Recall) and false alarm rates:
```bash
python ml/tune_threshold.py
```
*Output: `data/processed/rf_threshold_results.csv`*

---

## Benchmark Results (Validation Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **67.7%** | **0.292** | 0.629 | **0.399** | **0.721** | **0.396** |
| **XGBoost** | 63.2% | 0.265 | 0.657 | 0.378 | 0.704 | 0.388 |
| **Logistic Regression** | 53.7% | 0.222 | **0.687** | 0.336 | 0.656 | 0.336 |

---

## License

This project is licensed under the MIT License.
