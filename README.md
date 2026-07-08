🩺 Breast Cancer Prediction using Machine Learning

📌 Project Overview

This project predicts whether a breast tumor is Benign or Malignant using a Logistic Regression model trained on the Breast Cancer Wisconsin Dataset.
The application is built with Python, Scikit-Learn, and Streamlit.
📂 Project Structure

app.py
model.pkl
scaler.pkl
requirements.txt
README.md
breast_cancer_prediction.ipynb
data.csv
🚀 Features

Logistic Regression Model
StandardScaler for Feature Scaling
Interactive Streamlit UI
Real-time Prediction
Confidence Score Display
📊 Dataset

Total Samples: 569

Features: 30

Target:

0 → Benign
1 → Malignant
🛠 Technologies Used

Python
NumPy
Pandas
Scikit-Learn
Streamlit
Joblib
📈 Model Performance

Metric	Score
Accuracy	96.49%
Precision	97.50%
Recall	92.86%
F1 Score	95.12%
Confusion Matrix

[[71  1]
 [ 3 39]]
▶️ Installation

Clone the repository

git clone https://github.com/GopinathR07/Breast-Cancer-Prediction-Model
Move into the project

cd Breast-Cancer-Prediction
Install dependencies

pip install -r requirements.txt
Run the application

streamlit run app.py
🎯 Machine Learning Workflow

Data Loading
Exploratory Data Analysis (EDA)
Data Cleaning
Feature Selection
Label Encoding
Train-Test Split
Feature Scaling
Logistic Regression Training
Model Evaluation
Deployment with Streamlit
📌 Future Improvements

Random Forest Classifier
XGBoost Classifier
Hyperparameter Tuning
ROC-AUC Visualization
Model Comparison Dashboard
👨‍💻 Author

Gopinath R
Aspiring AI & Machine Learning Engineer
