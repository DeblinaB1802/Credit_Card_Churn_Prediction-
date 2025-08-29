# Credit Card Churn Prediction

## 📌 Project Overview
Customer churn, or attrition, is a significant challenge in the financial services industry. Retaining existing customers is often more cost-effective than acquiring new ones. In the credit card domain, churn directly impacts revenue streams, brand loyalty, and market competitiveness.  

This project focuses on **predicting credit card churn** using machine learning techniques. By analyzing customer demographics, account usage, and transaction behavior, we aim to identify at-risk customers and provide insights that enable targeted retention campaigns.  

---

## 🎯 Business Problem
A consumer credit card bank is struggling with customer attrition. The objectives are:  
- **Identify patterns** that lead to churn.  
- **Build predictive models** to classify whether a customer will churn.  
- **Provide actionable insights** to retain high-value customers.  

---

## 📊 Dataset Description
The dataset includes demographic and behavioral attributes of ~10,000 credit card customers.  

### Key Features
- **CLIENTNUM** – Unique identifier for each customer.  
- **Attrition_Flag** – Target variable (1 = churn, 0 = active).  
- **Customer_Age** – Age in years.  
- **Gender** – Male/Female.  
- **Dependent_count** – Number of dependents.  
- **Marital_Status** – Married/Single/Divorced.  
- **Income_Category** – Annual income category.  
- **Card_Category** – Type of credit card.  
- **Credit_Limit, Avg_Open_To_Buy** – Credit information.  
- **Total_Trans_Amt, Total_Trans_Ct** – Transaction behavior.  
- **Total_Relationship_Count** – Number of products/services with bank.  

---

## 🔬 Methodology

### 1. Data Preprocessing
- Handled missing values and irrelevant fields.  
- Encoded categorical variables (one-hot encoding, label encoding).  
- Scaled numerical features using StandardScaler/MinMaxScaler.  
- Checked class imbalance (churn vs non-churn).  

### 2. Exploratory Data Analysis (EDA)
- Distribution of churn across demographics.  
- Relationship between credit card usage and churn.  
- Correlation heatmaps to detect feature relationships.  
- Visualizations with Matplotlib/Seaborn.  

### 3. Model Development
Implemented and compared multiple supervised learning models:  
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting (XGBoost, LightGBM)  
- Support Vector Machines (optional)  
- Neural Networks (optional)  

### 4. Model Evaluation
- Train-Test split and cross-validation.  
- Metrics used:  
  - Accuracy  
  - Precision, Recall, F1-score  
  - ROC-AUC Curve  
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV.  

---

## 📈 Results & Insights
- Best-performing models: **Random Forest** and **Gradient Boosting**, with ROC-AUC ~0.85.  
- Key churn indicators:  
  - Customers with **low transaction counts/amounts**.  
  - Customers with **low credit utilization**.  
  - Customers with **fewer banking products**.  
- The model enables early intervention for high-risk customers.  

---

## ⚙️ How to Run the Project

### Prerequisites
Install required Python libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

### Steps to Execute
1. Clone the repository or download the notebook.  
2. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook Credit_card_churn_modelling.ipynb
   ```
3. Run cells sequentially to:  
   - Load and preprocess dataset  
   - Perform EDA  
   - Train and evaluate machine learning models  
   - Generate results and visualizations  

---

## 📂 Project Structure
```
├── Credit_card_churn_modelling.ipynb   # Main notebook
├── README.md                           # Project documentation
├── data/                               # Dataset folder (if provided)
├── models/                             # Saved ML models (optional)
└── results/                            # Evaluation reports & plots
```

---

## 🚀 Future Enhancements
- Use **SMOTE/ADASYN** for handling class imbalance.  
- Deploy the churn prediction model as a **Flask/FastAPI REST API**.  
- Create a **Streamlit dashboard** for interactive churn analytics.  
- Apply **Explainable AI tools (SHAP, LIME)** to interpret model predictions.  
- Incorporate **time-series analysis** of customer behavior.  

---

## 🏦 Business Value
- Enables **proactive customer retention**.  
- Reduces **customer acquisition costs**.  
- Increases **customer lifetime value (CLV)**.  
- Provides **data-driven insights** into churn drivers.  

---

 

---

