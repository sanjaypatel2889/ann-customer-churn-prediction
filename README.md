# ANN-Customer-Churn-Prediction
📊 Customer Churn Prediction using Artificial Neural Network (ANN)
📌 Project Overview:

Customer churn is a critical problem in the banking and telecom industries, where retaining existing customers is more cost-effective than acquiring new ones.
This project focuses on building a Customer Churn Prediction system using an Artificial Neural Network (ANN) to classify whether a customer is likely to leave the bank based on historical data.

The model is trained using structured customer data and optimized using GridSearchCV for hyperparameter tuning.

🎯 Objectives

Predict customer churn (Exited / Not Exited)

Perform proper data preprocessing and feature engineering

Build and train an ANN for binary classification

Optimize model performance using Grid Search

Follow industry-standard ML pipeline practices

🧠 Machine Learning Approach:

Problem Type: Binary Classification

Model Used: Artificial Neural Network (ANN)

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

🗂 Dataset Information:

Dataset: Churn_Modelling.csv

Target Variable: Exited

Key Features:

CreditScore

Geography

Gender

Age

Tenure

Balance

NumOfProducts

HasCrCard

IsActiveMember

EstimatedSalary

Note: Identifier columns like RowNumber, CustomerId, and Surname are removed during preprocessing.

🔄 Data Preprocessing Steps

Dropped irrelevant and identifier columns

Label Encoding for binary categorical features (Gender)

One-Hot Encoding for multi-class categorical features (Geography)

Feature scaling using StandardScaler

Train-Test split for model evaluation

⚙️ Model Architecture (ANN)

Input Layer (based on feature count)

One or more Hidden Layers (ReLU activation)

Output Layer with Sigmoid activation

Hyperparameters tuned using GridSearchCV:

Number of neurons

Number of hidden layers

Batch size

Epochs

🧪 Hyperparameter Tuning

Grid Search is applied using SciKeras + scikit-learn to find the best combination of ANN hyperparameters.

Why SciKeras?

Fully compatible with modern scikit-learn

Supports GridSearchCV without deprecated issues

Production-ready and actively maintained

🛠️ Technologies & Tools Used

Python 3.x

NumPy

Pandas

Scikit-learn

TensorFlow / Keras

SciKeras

Google Colab

Git & GitHub

📁 Project Structure
customer-churn-prediction-ann/
│
├── Sanjay.ipynb               # Main notebook
├── Churn_Modelling.csv        # Dataset
├── README.md                  # Project documentation

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/customer-churn-prediction-ann.git


Open the notebook in Google Colab or Jupyter Notebook

Install required dependencies

Run all cells sequentially

✅ Results

Achieved reliable accuracy for churn prediction

GridSearchCV successfully identified optimal ANN parameters

Model generalizes well on unseen data

📌 Key Learnings

Importance of proper data preprocessing

Correct handling of categorical features

ANN architecture tuning for structured data

Using SciKeras for sklearn compatibility

Debugging real-world ML pipeline errors

🔮 Future Improvements

Add precision, recall, and ROC-AUC metrics

Handle class imbalance using SMOTE

Deploy model using Streamlit or FastAPI

Save trained model and preprocessing pipeline

Convert notebook into a production ML pipeline

👨‍💻 Author

Sriram Sanjay
AI & Machine Learning Enthusiast
Focused on building practical, production-ready ML systems
