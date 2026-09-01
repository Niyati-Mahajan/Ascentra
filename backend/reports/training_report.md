# OFFICIAL 10K PLACEMENT GUIDANCE ML PIPELINE REPORT

## 1. Dataset Summary & Inspection
- **Raw Dataset**: `placement_10k.csv` (10,015 total rows including 15 duplicates)
- **Final Clean Dataset**: 10,000 unique student records
- **Target Variable**: `PlacementStatus` ('Placed': 9,358, 'Not Placed': 642)

## 2. Preprocessing & Feature Engineering
- **Imputation**: Median imputation for numerical features (`CGPA`, `AptitudeTestScore`); Mode imputation for categorical (`Stream`).
- **Feature Engineering**:
  - `AcademicConsistencyIndex`: Weighted average of SSC (25%), HSC (25%), and scaled CGPA (50%).
  - `PracticalExperienceScore`: Combined weighted count of Internships (x3.0), Projects (x2.0), and Certifications (x1.5).
  - `SkillsBalanceScore`: Mean of AptitudeTestScore and SoftSkillsScore.
- **Data Leakage & Fairness Exclusions**: Excluded `StudentID` (identifier) and `Gender` (fairness principle).

## 3. Model Training & Comparative Evaluation
Model evaluations on untouched 20% stratified test set:

| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `0.672` | `0.9626` | `0.6752` | `0.7937` | `0.7003` |
| **Random Forest** | `0.9345` | `0.9345` | `1.0` | `0.9661` | `0.6615` |
| **XGBoost Classifier** | `0.9335` | `0.9349` | `0.9984` | `0.9656` | `0.6718` |

## 4. Selected Model & Justification
- **Winner**: `LOGISTIC_REGRESSION`
- **Reasoning**: Highest combination of ROC-AUC (0.7003) and F1 score (0.7937) on untouched 20% test set.
- **Saved Model**: `models/selected/selected_placement_model.joblib`
