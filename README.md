# Diabetes Predictive Analysis

A comprehensive machine learning project that predicts the onset of diabetes based on diagnostic data. This project applies end-to-end data preprocessing, exploratory analysis, unsupervised learning, and classification models to uncover insights and improve predictive accuracy.

## Dataset

- **Source**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Samples**: 768
- **Features**: 8 medical predictors + 1 binary outcome (diabetes or not)

##  Objective

To predict the likelihood of diabetes in a patient using classification models based on medical features like glucose level, BMI, insulin, and more.

##  Technologies Used

- **Python**: Data analysis and modeling
- **Pandas, NumPy**: Data manipulation
- **Matplotlib, Seaborn**: Visualization
- **scikit-learn**: ML algorithms and preprocessing
- **XGBoost**: Advanced boosting classifier
- **UMAP, t-SNE, PCA**: Dimensionality reduction
- **Jupyter Notebook**: Development environment

## Project Workflow

### 1. Data Preprocessing
- Handled missing values using median imputation
- Normalized continuous features
- Selected important features (Glucose, BMI, Insulin)

### 2. Exploratory Data Analysis
- Correlation heatmaps and histograms
- Class imbalance analysis
- Visualization of class separability

### 3. Dimensionality Reduction
- Applied PCA, UMAP, and t-SNE to visualize latent structure in the dataset

### 4. Unsupervised Learning
- K-Means clustering into diabetic/non-diabetic groups
- Visualized clusters in 2D space using PCA

### 5. Supervised Learning
- **Logistic Regression**: Baseline accuracy 76.6%
- **Random Forest**: Improved to 81.2% with hyperparameter tuning
- **XGBoost**: Achieved 80.5% accuracy

### 6. Model Evaluation
- Accuracy, Precision, Recall, F1-score
- Feature importance plots
- Cross-validation

##  Key Insights

- **Glucose, BMI, and Insulin** are the most predictive features.
- Dimensionality reduction improves visualization and clustering separability.
- Ensemble models outperform linear models in capturing complex patterns.

##  Future Work

- Integrate neural networks and ensemble stacking
- Develop real-time web or mobile-based diabetes risk calculator
- Perform longitudinal analysis using time-series data

##  Repository Structure

Diabetes-Predictive-Analysis/
├── Diabetes Data Analysis.ipynb # Jupyter notebook with code and analysis
├── diabetes.csv # Dataset file
├── CSP571_FINALREPORT.pdf # Final report
├── DPA-Final Project-PPT.pptx # Project presentation slides
├── README.md # Project documentation
├── .gitignore # Ignored files