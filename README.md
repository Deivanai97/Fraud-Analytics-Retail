# 🛡️ Fraud Analytics System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Ready-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end Machine Learning web application designed to detect retail fraud and evaluate model performance.

---

## 📑 Table of Contents
- [Features](#-features)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#-usage)
- [Business Value](#-business-value)

---

## ✨ Features

The dashboard is separated into four distinct analytical tabs:

1. **📊 Exploratory Analysis:** Deep dive into dataset statistics, visualizing the severe class imbalance, spending distributions, and bivariate relationships between customer behavior and fraud.
2. **🎯 Model Evaluation:** Translates raw ML metrics (Accuracy, Precision, Recall, PR-AUC) into business impact. Visualizes the precision-recall tradeoff to balance catching fraud against minimizing false alarms (customer friction).
3. **🧠 Global Risk Drivers:** Leverages SHAP (SHapley Additive exPlanations) to explain the global logic of the AI, showing exactly which features drive fraud predictions across the entire dataset.
4. **🔍 Live Inspector:** An operational tool for fraud analysts. Select a specific transaction to instantly view the AI's verdict/probability and examine a local SHAP waterfall chart detailing the exact reasons for the alert.

---

## 🛠️ Tech Stack

- **Frontend Interface:** [Streamlit](https://streamlit.io/)
- **Machine Learning Backend:** [LightGBM](https://lightgbm.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
- **Explainable AI (XAI):** [SHAP](https://shap.readthedocs.io/)
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure

```text
fraud_detection_project/
│
├── backend.py                        # Data pipeline, feature engineering, and ML model training
├── app.py                            # Streamlit application and UI layout
├── retail_fraud_detection_100k.csv   # Dataset (100,000 transaction records)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## ⚙️ Installation

To run this project locally, ensure you have Python 3.8+ installed

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/fraud-analytics-dashboard.git
cd fraud-analytics-dashboard
```

**2. Create a virtual environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate  
# On Windows use  venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Once the dependencies are installed, you can launch the dashboard using Streamlit:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 💡 Business Value

- **Explainable AI (XAI):** Moves beyond "black box" predictions. By utilizing SHAP values, stakeholders can trust the model and deeply understand its decision-making criteria.
- **Operational Efficiency:** The "Live Inspector" accelerates manual review times for fraud investigators by summarizing exactly which customer behaviors look suspicious.
- **Optimized for Real-World Data:** Prioritizes PR-AUC over standard Accuracy, recognizing that in highly imbalanced fraud datasets, minimizing false positives (customer friction) is just as critical as maximizing true positives (fraud caught).

---
## 📝 Acknowledgements & Credits

* **Development & Architecture:** Built as a portfolio project to demonstrate full-stack machine learning workflows, data analytics, and interactive dashboard design.
* **AI Assistance:** Leveraged AI tools for collaborative code structuring, and debugging assistance.