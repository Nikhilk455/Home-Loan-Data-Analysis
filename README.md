# 🏠 Home Loan Approval Prediction using Deep Learning

This project focuses on predicting home loan approval using a deep learning model built with TensorFlow/Keras. The workflow includes data cleaning, preprocessing, encoding, feature scaling, model training, and evaluation using multiple performance metrics.

📊 Project Overview

The goal of this project is to analyze a home loan dataset and build a predictive model that determines whether a loan application will be approved. The project covers:

- Data loading & inspection

- Data cleaning (handling missing values)

- One-hot encoding of categorical variables

- Train–test split

- Feature scaling using StandardScaler

- Building a neural network model with TensorFlow/Keras

- Model training & validation

- Performance evaluation (sensitivity, specificity, AUC-ROC)

🛠️ Tech Stack & Libraries

- Python

- Pandas

- NumPy

- Scikit-learn

- TensorFlow / Keras

Matplotlib / Seaborn (if used for EDA)

🔧 Key Steps in the Workflow
1. Data Loading

        df = pd.read_csv('loan_data.csv')

2. Handling Missing Values

- Dropping rows with missing values

- Optionally replacing with mean/median if required

3. One-Hot Encoding

        df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

4. Train–Test Split

        X = df_encoded.drop('TARGET', axis=1)
        y = df_encoded['TARGET']

5. Feature Scaling

Standardizing numerical features:

    scaler = StandardScaler()

6. Deep Learning Model

A simple feedforward neural network:

    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
7. Model Evaluation

Metrics used:

- Sensitivity (Recall)

- Specificity

- AUC-ROC

- Confusion Matrix

- Predictions with threshold = 0.5

📈 Model Performance

The model predicts loan approval probabilities and evaluates the classification quality using:

- Recall: How well the model identifies approved loans

- Specificity: How well it identifies rejected loans

- ROC-AUC: Overall prediction strength

🎯 Outcome

This project provides a complete machine-learning pipeline for home loan approval prediction, showcasing:

✔ Data preprocessing
✔ Deep learning model building
✔ Imbalanced class handling with class weights
✔ Evaluation with practical metrics

It can be extended for:

- Hyperparameter tuning

- Adding more layers/neuron optimization

- Deployment via Flask/Streamlit

🔧 Key Steps in the Workflow
1. Data Loading
