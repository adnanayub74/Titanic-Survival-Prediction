# Titanic Survival Prediction: Machine Learning Pipeline

This project implements a complete machine learning pipeline to predict survival on the Titanic, following a structured methodology of preprocessing, hyperparameter tuning, and comparative model evaluation.

**Objectives**

* Implement a robust **scikit-learn pipeline** to combine data preprocessing and modeling.
* Use **Grid Search Cross-Validation** to optimize hyperparameters for classification.
* Compare the performance and feature interpretability of two distinct models:
    * **Random Forest Classifier** (Ensemble)
    * **Logistic Regression** (Linear)

**Methodology Highlights**
| Step | Technique Used | Purpose |
| :--- | :--- | :--- |
| **Data Split** | Stratification (`stratify=y`) | Ensure the class imbalance is maintained in training and testing sets. |
| **Preprocessing** | `ColumnTransformer` | Apply **`StandardScaler`** and median imputation to numerical features. Apply **`OneHotEncoder`** and mode imputation to categorical features. |
| **Tuning** | `GridSearchCV` | Systematically find the best hyperparameters for both models. |

**Final Results**

The models were evaluated on the unseen test set after cross-validation.

| Model | Model Type | Test Set Accuracy |
| :--- | :--- | :--- |
| **Random Forest** (Ensemble) | Ensemble (Bagging) | **0.832402** (83.24%) |
| **Logistic Regression** | Linear Classifier | **0.826816** (82.68%) |

**Key Conclusion**
The **Random Forest Classifier** (your ensemble model) performed marginally better. The analysis showed that while the overall accuracy was nearly identical, the features the models relied on for prediction were very different, suggesting underlying **collinearity** in the dataset.

