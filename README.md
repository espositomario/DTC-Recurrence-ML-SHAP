# Differentiated Thyroid Cancer (DTC) Recurrence Prediction and SHAP Analysis


**Authors**: [Irene D'Onofrio](https://github.com/irenedonofrio) and [Mario Esposito](https://github.com/espositomario)

**Aim**: This project provides a robust benchmarking of machine learning models for predicting differentiated thyroid cancer (DTC) recurrence, using nested cross-validation (nCV) for evaluation and SHAP analysis for interpretation.

**Methods**: Models tested included Support Vector Machine, XGBoost, Random Forest, Decision Trees, Logistic Regression, and Multi-Layer Perceptron, evaluated using a stratified 5-fold outer nCV with a 3-fold inner loop for hyperparameter tuning. SHAP analysis was applied to the best-performing model (SVM) to assess feature importance and explain predictions.

**Results**: SVM, XGBoost, and RF showed the strongest generalization, with SVM achieving the highest average MCC of 0.91 ± 0.04. SHAP analysis identified "Response" as the most influential feature (followed by others), and provided insight into misclassified cases.

### DTC Dataset

UC Irvine Machine Learning Repository: [DTC Dataset](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)

- **Donated:** 10/30/2023
- **Description:** 13 clinicopathologic features collected over 15 years, with a minimum 10-year follow-up per patient.
- **Dataset Characteristics:** Tabular
- **Primary Task:** Binary Classification
- **Target label**: Recurred/Not Recurred
- **Instances:** 383
- **Suggested split**: No
- **Features:** Age, Gender, Smoking, Hx Smoking, Hx Radiotherapy,
Thyroid Function, Physical Examination, Adenopathy, Pathology,
Focality, Risk, T, N, M, Stage, Response
- **Reference**: [Springer Link](https://link.springer.com/article/10.1007/s00405-023-08299-w#Sec2)






### Jupyter notebook Table of Contents
- Exploratory Data Analysis  
  - Data downloading  
  - Order categories (Ordinal features)  
  - Plot Features Distributions  
  - Plot Feature Distributions stratified per classes (Recurred / Not Recurred)  
  - Feature Encoding  
- Nested Cross-Validation (nCV)  
  - Models hyperparameters space  
  - Stratified 5-fold nCV (3-fold inner CV)  
  - Save or Import existing nCV_results  
  - Compare models metrics on testing  
    - MCC, ROC AUC and PRC AUC  
    - ROC and PRC curves  
- SHAP analysis on SVM  
  - SHAP on testing data (loop in the outer CV)  
  - Save or Import existing results  
  - SHAP Visualization  
    - Global feature importance  
    - SHAP values per features (sample-wise)  
    - Feature values effect on prediction (sorted by average feature importance)  
  - Misclassified samples  

### Other Studies Using the Same Dataset

1. [MDPI Paper](https://www.mdpi.com/2673-9585/4/4/29)
2. [arXiv Paper](https://arxiv.org/abs/2410.10907)
3. [Springer Link](https://link.springer.com/article/10.1007/s00405-023-08299-w#Sec2)