# 🎓 Student Performance Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Predicting student academic success using Machine Learning. This project analyzes various academic and lifestyle factors to estimate a student's **Performance Index**.

## 🚀 Overview

This repository contains a complete end-to-end Machine Learning pipeline, from data exploration and preprocessing to model deployment. We use a **Linear Regression** model to predict performance scores based on features like study hours, previous grades, and sleep patterns.

### Key Features
- **Exploratory Data Analysis (EDA)**: Comprehensive visualization of data distributions and correlations.
- **Preprocessing Pipeline**: Standard scaling and Principal Component Analysis (PCA) for optimal feature representation.
- **Interactive Dashboard**: A Streamlit-based web application for real-time predictions.

## 📊 Dataset

The dataset used is `Student_Performance.csv`, which includes:
- **Hours Studied**: Number of hours a student spends studying.
- **Previous Scores**: Scores from previous academic tests.
- **Extracurricular Activities**: Participation in activities outside the curriculum (Yes/No).
- **Sleep Hours**: Average daily sleep duration.
- **Sample Question Papers Practiced**: Number of mock papers attempted.
- **Performance Index**: The target variable (Scale 10-100).

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Biswajeet111/Student-Performance-ML-Project.git
   cd Student-Performance-ML-Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Training & Analysis
Open the Jupyter Notebook to see the model training process:
```bash
jupyter notebook Student_Performance.ipynb
```

### 2. Run the Web App
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

## 📈 Model Results

The model achieves high accuracy on the test set:
- **Mean Absolute Error (MAE)**: ~1.61
- **R² Score**: ~0.98

## 🧱 Project Structure
```text
.
├── Student_Performance.csv     # Raw dataset
├── Student_Performance.ipynb   # EDA & Model Training
├── app.py                      # Streamlit Frontend
├── student_performance_model.pkl # Trained Model
├── scaler.pkl                  # Feature Scaler
├── pca.pkl                     # PCA Transformer
└── requirements.txt            # Project Dependencies
```

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
Developed with ❤️ by [Biswajeet Kumar](https://github.com/Biswajeet111)
