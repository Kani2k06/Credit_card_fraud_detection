import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection Using Machine Learning")
st.markdown("This Streamlit application detects fraudulent credit card transactions using Machine Learning.")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# ------------------------------
# LOAD DATA
# ------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    st.subheader("📌 Dataset Information")
    st.write("Shape:", data.shape)
    st.write("Columns:", data.columns.tolist())

    # ------------------------------
    # CHECK TARGET COLUMN
    # ------------------------------
    if 'Class' not in data.columns:
        st.error("Dataset must contain a 'Class' column where 0 = Normal and 1 = Fraud")
    else:
        # ------------------------------
        # DATA PREPROCESSING
        # ------------------------------
        X = data.drop('Class', axis=1)
        y = data['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ------------------------------
        # MODEL TRAINING
        # ------------------------------
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # ------------------------------
        # PREDICTIONS
        # ------------------------------
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # ------------------------------
        # RESULTS
        # ------------------------------
        st.subheader("✅ Model Performance")
        st.success(f"Accuracy: {accuracy * 100:.2f}%")

        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # ------------------------------
        # CONFUSION MATRIX
        # ------------------------------
        st.subheader("📉 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

        # ------------------------------
        # MANUAL PREDICTION
        # ------------------------------
        st.subheader("🔍 Manual Transaction Prediction")
        st.write("Enter transaction values manually to predict fraud.")

        input_data = []

        for column in X.columns:
            value = st.number_input(f"Enter {column}", value=0.0)
            input_data.append(value)

        if st.button("Predict Transaction"):
            prediction = model.predict([input_data])[0]

            if prediction == 1:
                st.error("⚠ Fraudulent Transaction Detected")
            else:
                st.success("✅ Legitimate Transaction")

else:
    st.info("Please upload a CSV dataset to begin.")
