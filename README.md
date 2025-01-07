# Automated Report Generator

## Overview
The **Automated Report Generator** is a Streamlit-based web application that simplifies the process of data analysis and machine learning. Users can upload a dataset, perform exploratory data analysis (EDA), preprocess the data, train a machine learning model, and generate a detailed PDF report with visualizations and metrics—all in one streamlined workflow.

---

## Features
1. **Dataset Handling**:
   - Load preset datasets (e.g., Iris, Wine, Diabetes).
   - Upload custom datasets as CSV files.
   - Load datasets directly from URLs.

2. **Exploratory Data Analysis (EDA)**:
   - Automated generation of distribution plots, count plots, and histograms.
   - Correlation heatmap to understand feature relationships.
   - Pairplot visualizations for in-depth feature interactions.

3. **Data Preprocessing**:
   - Handles missing values and scales numeric features.
   - Encodes categorical target columns for classification tasks.
   - Determines whether the problem is classification or regression based on the target column.

4. **Model Training**:
   - Automatic selection of Random Forest models for both classification and regression tasks.
   - Hyperparameter tuning using GridSearchCV.
   - Outputs key metrics such as accuracy, confusion matrix, precision-recall AUC, and regression errors.

5. **Report Generation**:
   - Compiles EDA visuals and model evaluation metrics into a well-structured PDF.
   - Includes a feature importance chart for deeper insights into the trained model.

6. **Interactive Streamlit Interface**:
   - Easy-to-use sidebars for dataset selection and configuration.
   - Downloadable PDF report with all results and visualizations.

---

## Requirements
### Python Libraries
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn` (Scikit-learn)
- `fpdf`
- `joblib`
- `logging`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RohanSai22/datavizualizer
   cd automated-report-generator
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## How to Use
1. **Load Dataset**:
   - Select a dataset from the sidebar options: Preset, Upload, or URL.
   - If uploading a CSV file, ensure it is well-formatted and contains the required data.

2. **View Dataset Overview**:
   - Preview the first few rows of the dataset in the main window.

3. **Select Target Column**:
   - Choose the column you want to predict from the dropdown menu.

4. **Generate Report**:
   - Click the "Generate Report" button to start the analysis.
   - Download the PDF report once it is ready.

---

## File Structure
```
.
├── streamlit_app.py    # Main Streamlit application script
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation

```

---

## Example Usage
1. **Preset Dataset**:
   - Select "Iris" from the sidebar, set the target column to `species`, and generate a classification report.
   
2. **Custom Dataset**:
   - Upload a CSV file containing housing prices, select the `price` column as the target, and generate a regression report.

---

## Features Roadmap
- Add support for more model types (e.g., SVM, Neural Networks).
- Enable advanced visualizations like SHAP or LIME for explainability.
- Extend the report with additional sections such as detailed model comparison.
- Optimize application performance for large datasets.

---

## Contribution Guidelines
1. Fork the repository and create a feature branch.
2. Make your changes and submit a pull request.
3. Ensure your code passes the necessary tests and adheres to the project style guide.

---

## License
This project is licensed under the MIT License.

---

## Author
Developed by **Rohan**. If you have any questions or feedback, feel free to reach out.
