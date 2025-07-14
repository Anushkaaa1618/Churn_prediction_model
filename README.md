# Churn_prediction_model

This project builds a machine learning model to predict customer churn using a dataset of bank customers. It uses XGBoost, a powerful gradient boosting algorithm, to classify whether a customer is likely to leave the bank.

## 📁 Dataset

The dataset contains information about 10,000 bank customers, including:
- Demographic data (Age, Gender, Geography)
- Financial details (Balance, Credit Score, Estimated Salary)
- Bank behavior (Tenure, Products, Card and Membership info)

The target column is:
- `Exited`: 1 if the customer has churned, 0 otherwise.

## 🔧 Features Used

- CreditScore  
- Age  
- Tenure  
- Balance  
- NumOfProducts  
- HasCrCard  
- IsActiveMember  
- EstimatedSalary  
- Gender (encoded)  
- Geography (one-hot encoded)

## 📊 Model

- **Algorithm:** XGBoost Classifier  
- **Preprocessing:** Label Encoding, One-Hot Encoding, Feature Scaling (StandardScaler)  
- **Evaluation:** Accuracy Score & Classification Report  

## 🧪 Results

The model achieved good accuracy in identifying churned customers and provides a detailed classification report.

## 🚀 How to Run

1. Clone the repository  
2. Make sure Python 3 and required packages are installed:
3. Run the script:
   
## 📌 File Structure

- `Churn_Modelling.csv` – The dataset
- `churn_prediction.py` – Main model training script
- `README.md` – Project description

## ✅ Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Model deployment (Flask/Streamlit web app)
- Feature importance visualization

---




