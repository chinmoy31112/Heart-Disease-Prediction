# Heart Disease Prediction using Machine Learning

This project focuses on predicting heart disease in patients based on various medical attributes using various Machine Learning classification algorithms.

## Project Overview
Heart disease is one of the leading causes of death worldwide. This project uses a dataset of patient medical records to build a predictive model that can accurately identify whether a person has heart disease or not.

## Dataset Features
The dataset (`heart.csv`) contains 297 entries with 14 attributes:
1.  **Age**: Age of the patient
2.  **Sex**: (1 = male; 0 = female)
3.  **Chest Pain Type (cp)**: (0, 1, 2, 3)
4.  **Trestbps**: Resting blood pressure
5.  **Chol**: Serum cholesterol in mg/dl
6.  **Fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7.  **Restecg**: Resting electrocardiographic results
8.  **Thalach**: Maximum heart rate achieved
9.  **Exang**: Exercise induced angina (1 = yes; 0 = no)
10. **Oldpeak**: ST depression induced by exercise relative to rest
11. **Slope**: The slope of the peak exercise ST segment
12. **CA**: Number of major vessels (0-3)
13. **Thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
14. **Target**: 0 = No Heart Disease, 1 = Heart Disease

## Technologies Used
- **Language**: Python
- **Libraries**:
    - `Pandas` & `NumPy`: For data manipulation and analysis.
    - `Matplotlib` & `Seaborn`: For data visualization and EDA.
    - `Scikit-learn`: For model building, training, and evaluation.

## Machine Learning Models Implemented
The project explores and compares several classification models:
- **Logistic Regression**: (Best performing model with ~85% accuracy)
- **K-Nearest Neighbors (KNN)**: (~76% accuracy)
- **Support Vector Machine (SVM)**: (~74% accuracy with Linear Kernel)
- **Naive Bayes**: (~72% accuracy)

## Workflow
1.  **Data Collection**: Loading the `heart.csv` dataset.
2.  **Exploratory Data Analysis (EDA)**: Understanding data distributions and correlations using heatmaps and histograms.
3.  **Data Preprocessing**: Handling missing values, feature scaling (StandardScaler/MinMaxScaler), and splitting the data into training and testing sets.
4.  **Model Training**: Training various classifiers on the processed data.
5.  **Evaluation**: Comparing accuracy scores and confusion matrices.
6.  **Prediction System**: A built-in system to predict heart disease for new patient data inputs.

## How to use
1. **Prerequisites**: Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
2. **Jupyter Notebook**: Run the `Heart_Disease_Prediction.ipynb` notebook to see the full data analysis and model training steps.
3. **Command Line (Fast Run)**: To quickly run the model and see predictions for sample patients, use the provided script:
   ```bash
   .\.venv\Scripts\python.exe run_prediction.py
   ```
   The results will be displayed in the terminal and saved to `results.txt`.
