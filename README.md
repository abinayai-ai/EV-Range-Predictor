# ⚡ EV Range Intelligence – Electric Vehicle Range Predictor

A Machine Learning-based application that predicts the real-world driving range of Electric Vehicles (EVs) using vehicle specifications and driving conditions.

## 🚗 Project Overview

EV Range Intelligence helps estimate how far an electric vehicle can travel based on factors such as battery capacity, motor power, vehicle weight, driving style, traffic conditions, and ambient temperature.

The application uses a trained Random Forest model and provides an interactive dashboard built with Streamlit.

## ✨ Features

- Predict EV real-world driving range.
- Interactive Streamlit dashboard.
- Vehicle specifications and driving condition inputs.
- Data analytics and visualizations.
- Manufacturer-wise range comparison.
- ARAI certified range vs estimated real-world range comparison.
- Machine Learning model evaluation.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 🤖 Machine Learning Models

Two regression models were evaluated:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 44.97 | 56.69 | 0.8433 |
| Random Forest | 19.90 | 29.92 | 0.9563 |

**Best Model: Random Forest**

## 📂 Project Structure

```text
EV-Range-Predictor/
├── dataset/
│   └── india_ev_range_dataset.csv
├── models/
│   └── ev_range_model.pkl
├── notebooks/
├── app.py
├── main.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/abinayai-ai/EV-Range-Predictor.git
```

### 2. Navigate to the project folder

```bash
cd EV-Range-Predictor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```bash
.venv\Scripts\Activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Dataset

Dataset: India EV Range Dataset

Source: [Hugging Face – India EV Range Dataset](https://huggingface.co/datasets/pjjg163/india-ev-range-dataset)

The dataset contains vehicle specifications and estimated driving scenarios. Predictions should be treated as estimates, not guaranteed real-world vehicle performance.

## 🎯 Project Objective

To develop an interactive Machine Learning application that estimates EV driving range and helps users understand how vehicle specifications and driving conditions influence range.

## 👩‍💻 Author

**Abinaya J**

B.Tech – Artificial Intelligence and Data Science

GitHub: [abinayai-ai](https://github.com/abinayai-ai)

## 🚀 Live Demo

[Click here to try EV Range Predictor](https://abinayai-ai-ev-range-predictor-app-cjxx1k.streamlit.app/)

LinkedIn: [Abinaya J](https://www.linkedin.com/in/abinaya-j-56155a327/)

---

⭐ If you find this project interesting, consider giving the repository a star!
