"# ANN-Classification-Churn-Prediction" 

# 🏦 Bank Churn Prediction using Artificial Neural Networks (ANN)

A deep learning-based classification project to predict whether a bank customer is likely to **churn (leave the bank)** or **stay**, built and deployed with TensorFlow and Streamlit.

---

## 🚀 Project Overview
This project predicts customer churn based on demographic and account activity data using an **Artificial Neural Network (ANN)**.  
The aim is to help banks identify customers at risk of leaving and take preventive measures to improve retention.

---

## 🧠 Model Development

### ⚙️ Key Steps
- Implemented an **Artificial Neural Network (ANN)** using **TensorFlow** and **Keras**.
- Performed **data preprocessing** including:
  - Label Encoding for categorical columns
  - One-Hot Encoding for geographic data
  - Feature scaling using `StandardScaler`
- Split data into training and testing sets for model evaluation.

### 🔍 Hyperparameter Tuning
- Used **GridSearchCV** with **SciKerasClassifier** to tune key parameters:
  - Number of **hidden layers**
  - Number of **neurons per layer**
  - **Batch size**
  - **Epochs**
  - **Optimizer (Adam / RMSprop / SGD)**
- Achieved significant performance improvement — training accuracy increased from **80% → 85.89%**.

### 🧩 Model Architecture
| Layer | Type | Activation | Notes |
|-------|------|-------------|--------|
| Input | Dense | ReLU | Input features after encoding |
| Hidden 1 | Dense | ReLU | Tuned neuron count |
| Hidden 2 | Dense | ReLU | Tuned neuron count |
| Output | Dense | Sigmoid | Binary classification output |

---

## 📊 Performance Metrics

| Metric | Train Accuracy | Validation Accuracy |
|:-------:|:---------------:|:-------------------:|
| **Accuracy** | **0.8679 (~86.7%)** | **0.8565 (~85.6%)** |

Training tracked using **TensorBoard** for better visualization of epoch-wise accuracy and loss.

---

## 🧰 Tech Stack

- **Python 3.10+**
- **TensorFlow 2.20.0**
- **Keras / SciKeras**
- **Pandas, NumPy**
- **Scikit-learn**
- **TensorBoard**
- **Streamlit**
- **Plotly**

---

## 💻 Streamlit Web App

- Developed an **interactive Streamlit dashboard** for live predictions.
- Users can input customer details and get instant churn predictions.
- Custom dark theme used for enhanced UI/UX.

To run:
```bash
streamlit run app.py
