import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fpdf import FPDF
import os
import tempfile
import json
import logging
from datetime import datetime
import joblib

# Set up logging
logging.basicConfig(
    filename="app_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_message(level, message):
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    elif level == "warning":
        logging.warning(message)

# Streamlit App Definition
def main():
    st.title("Advanced Automated Machine Learning Report Generator")

    st.sidebar.title("Dataset Input")

    dataset_option = st.sidebar.selectbox(
        "Choose Dataset Source:",
        ("Select a Preset Dataset", "Upload Your Dataset", "Enter Dataset URL")
    )

    df = None

    try:
        if dataset_option == "Select a Preset Dataset":
            preset_dataset = st.sidebar.selectbox(
                "Choose Preset Dataset:",
                ("Iris", "Wine", "Diabetes")
            )
            df = load_preset_dataset(preset_dataset)

        elif dataset_option == "Upload Your Dataset":
            uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)

        elif dataset_option == "Enter Dataset URL":
            dataset_url = st.sidebar.text_input("Enter Dataset URL:")
            if dataset_url:
                df = pd.read_csv(dataset_url)

        if df is not None:
            st.sidebar.success("Dataset Loaded Successfully")
            st.write("### Dataset Overview")
            st.dataframe(df.head())

            target_column = st.sidebar.selectbox("Select Target Column:", df.columns)

            if st.sidebar.button("Generate Report"):
                generate_report(df, target_column)
    except Exception as e:
        log_message("error", f"Error during dataset processing: {e}")
        st.error(f"An error occurred: {e}")

def load_preset_dataset(preset_name):
    try:
        if preset_name == "Iris":
            from sklearn.datasets import load_iris
            data = load_iris(as_frame=True)
        elif preset_name == "Wine":
            from sklearn.datasets import load_wine
            data = load_wine(as_frame=True)
        elif preset_name == "Diabetes":
            from sklearn.datasets import load_diabetes
            data = load_diabetes(as_frame=True)

        df = pd.concat([data.data, data.target], axis=1)
        df.columns = list(data.feature_names) + ["target"]
        return df
    except Exception as e:
        log_message("error", f"Error loading preset dataset {preset_name}: {e}")
        raise

def preprocess_data(df, target_col):
    try:
        df = df.dropna()

        task_type = "classification" if df[target_col].nunique() <= 20 else "regression"
        label_encoders = {}

        if task_type == "classification":
            if df[target_col].dtype == 'object':
                le = LabelEncoder()
                df[target_col] = le.fit_transform(df[target_col])
                label_encoders[target_col] = le

        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference([target_col])
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        log_message("info", f"Data preprocessing completed for {task_type}.")
        return df, label_encoders, scaler, task_type
    except Exception as e:
        log_message("error", f"Error during preprocessing: {e}")
        raise

def perform_eda(df):
    plots = []

    try:
        for col in df.columns:
            plt.figure()
            if df[col].nunique() < 20:
                sns.countplot(x=col, data=df)
            else:
                sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name)
            plots.append(temp_file.name)
            plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name)
        plots.append(temp_file.name)
        plt.close()

        sns.pairplot(df)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name)
        plots.append(temp_file.name)
        plt.close()

        log_message("info", "EDA completed.")
    except Exception as e:
        log_message("error", f"Error during EDA: {e}")
        raise

    return plots

def train_model(df, target_col, task_type):
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task_type == "classification":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
            }
        else:
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
            }

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy' if task_type == "classification" else 'neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        if task_type == "classification":
            precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
            pr_auc = auc(recall, precision)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'precision_recall_auc': pr_auc
            }
        else:
            metrics = {
                'mean_squared_error': mean_squared_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred),
                'mean_absolute_error': np.mean(np.abs(y_test - y_pred))
            }

        joblib.dump(best_model, os.path.join(tempfile.gettempdir(), "trained_model.pkl"))

        log_message("info", "Model training completed.")
        return best_model, metrics
    except Exception as e:
        log_message("error", f"Error during model training: {e}")
        raise

def generate_report(df, target_col):
    try:
        processed_df, _, _, task_type = preprocess_data(df, target_col)
        eda_plots = perform_eda(processed_df)
        model, metrics = train_model(processed_df, target_col, task_type)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14, style='B')
        pdf.cell(200, 10, txt="Advanced Machine Learning Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Exploratory Data Analysis", ln=True, align='L')
        for plot_path in eda_plots:
            pdf.add_page()
            pdf.image(plot_path, x=10, y=10, w=190)

        pdf.add_page()
        pdf.cell(200, 10, txt=f"Model Evaluation Metrics ({task_type.capitalize()})", ln=True, align='L')
        pdf.set_font("Courier", size=10)
        if task_type == "classification":
            pdf.multi_cell(0, 10, txt=f"Accuracy: {metrics['accuracy']}")
            pdf.multi_cell(0, 10, txt=f"Precision-Recall AUC: {metrics['precision_recall_auc']:.2f}")
            pdf.multi_cell(0, 10, txt="Classification Report:")
            pdf.multi_cell(0, 10, txt=json.dumps(metrics['classification_report'], indent=2))
        else:
            pdf.multi_cell(0, 10, txt=f"Mean Squared Error: {metrics['mean_squared_error']:.2f}")
            pdf.multi_cell(0, 10, txt=f"R2 Score: {metrics['r2_score']:.2f}")
            pdf.multi_cell(0, 10, txt=f"Mean Absolute Error: {metrics['mean_absolute_error']:.2f}")

        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Advanced Visualizations and Analysis", ln=True, align='L')

        # Add advanced visualization: Feature Importances
        if task_type == "classification" or task_type == "regression":
            feature_importances = model.feature_importances_
            feature_names = processed_df.drop(columns=[target_col]).columns
            sorted_idx = np.argsort(feature_importances)[::-1]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importances[sorted_idx], y=feature_names[sorted_idx], palette="viridis")
            plt.title("Feature Importance")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name)
            plt.close()

            pdf.add_page()
            pdf.cell(200, 10, txt="Feature Importance Plot", ln=True, align='L')
            pdf.image(temp_file.name, x=10, y=30, w=190)

        # Save PDF Report
        report_path = os.path.join(tempfile.gettempdir(), f"ML_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(report_path)

        log_message("info", f"Report generated and saved at {report_path}")
        st.success("Report generated successfully!")
        with open(report_path, "rb") as f:
            st.download_button(
                label="Download Report",
                data=f,
                file_name="ML_Report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        log_message("error", f"Error during report generation: {e}")
        st.error(f"An error occurred while generating the report: {e}")

if __name__ == "__main__":
    main()