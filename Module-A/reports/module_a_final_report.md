# ASCENTRA Module A Technical Report

## Dataset And Target
- Rows: 4424
- Columns: 16
- Target column: `academic_risk`
- Class distribution: {'0': 2557, '1': 1867}
- Class percentage: {'0': 57.8, '1': 42.2}
- Target generation: The refined-data notebook samples academic_risk from a synthetic probabilistic risk score using attendance, best_2_ca_average, mid_term_score, previous_semester_tgpa, academic_trend, and noise.
- Synthetic-target limitation: The target is generated from several predictor variables. This is acceptable for the current synthetic dataset exercise but limits real-world claims until validated against institutional outcomes.

## Leakage And Redundancy
- `student_id` and target are excluded from features.
- Preprocessing, model search, and threshold tuning use training data only.
- Semester 1 previous TGPA remains missing until pipeline imputation; it is not replaced by 0.
- High-correlation feature pairs >= 0.85: [{'feature_1': 'best_2_ca_average', 'feature_2': 'ca_average', 'absolute_correlation': 0.9869297544794006}, {'feature_1': 'mid_term_score', 'feature_2': 'attendance_midterm_interaction', 'absolute_correlation': 0.9540667802907985}, {'feature_1': 'best_2_ca_average', 'feature_2': 'attendance_best2_interaction', 'absolute_correlation': 0.931498285463124}, {'feature_1': 'ca_average', 'feature_2': 'attendance_best2_interaction', 'absolute_correlation': 0.924242919124276}, {'feature_1': 'ca1_score', 'feature_2': 'ca_average', 'absolute_correlation': 0.9083390437073594}, {'feature_1': 'best_2_ca_average', 'feature_2': 'tgpa_best2_interaction', 'absolute_correlation': 0.9046931658200354}, {'feature_1': 'ca_average', 'feature_2': 'tgpa_best2_interaction', 'absolute_correlation': 0.897545917859285}, {'feature_1': 'attendance_best2_interaction', 'feature_2': 'tgpa_best2_interaction', 'absolute_correlation': 0.8864596668158042}]
- Mid-term comparison: [{'variant': 'A_with_mid_term', 'accuracy_mean': 0.624748479690584, 'accuracy_std': 0.022434446532862515, 'precision_mean': 0.549784291678031, 'recall_mean': 0.6144486094588224, 'f1_mean': 0.5803028838292242, 'f1_std': 0.02458329291294222, 'roc_auc_mean': 0.6670710346028527, 'roc_auc_std': 0.030902525420926325}, {'variant': 'B_without_mid_term', 'accuracy_mean': 0.6255963368733969, 'accuracy_std': 0.02654390662107679, 'precision_mean': 0.5513360971900346, 'recall_mean': 0.6144598325514579, 'f1_mean': 0.5810773897338292, 'f1_std': 0.025633451295144424, 'roc_auc_mean': 0.6656555254742524, 'roc_auc_std': 0.029900310188860007}]

## Feature Engineering
- Final model features: `semester`, `attendance_percentage`, `ca1_score`, `ca2_score`, `ca3_score`, `best_2_ca_average`, `mid_term_score`, `previous_semester_tgpa`, `ca_average`, `ca_improvement`, `ca_consistency`, `attendance_best2_interaction`, `attendance_midterm_interaction`, `tgpa_best2_interaction`, `previous_tgpa_available`, `academic_trend`
- Engineered features: `ca_average`, `ca_improvement`, `ca_consistency`, `attendance_best2_interaction`, `attendance_midterm_interaction`, `tgpa_best2_interaction`, `previous_tgpa_available`

## Cross-Validation Model Comparison
| Model | Accuracy | Accuracy SD | Precision | Recall | F1 | F1 SD | ROC-AUC | ROC-AUC SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6332 | 0.0207 | 0.5578 | 0.6325 | 0.5928 | 0.0238 | 0.6884 | 0.0326 |
| XGBoost Tuned | 0.6315 | 0.0219 | 0.5562 | 0.6325 | 0.5917 | 0.0234 | 0.6777 | 0.0328 |
| XGBoost Baseline | 0.6247 | 0.0224 | 0.5498 | 0.6144 | 0.5803 | 0.0246 | 0.6671 | 0.0309 |
| Random Forest | 0.6174 | 0.0231 | 0.5461 | 0.5576 | 0.5518 | 0.0250 | 0.6568 | 0.0304 |
| HistGradientBoostingClassifier | 0.6174 | 0.0247 | 0.5578 | 0.4578 | 0.5028 | 0.0282 | 0.6409 | 0.0271 |

## Final Model
- Algorithm: tuned XGBoost binary classifier
- Classification threshold: 0.39
- Hyperparameters: `{'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': 1.0, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': True, 'eval_metric': 'logloss', 'feature_types': None, 'feature_weights': None, 'gamma': 0, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.05, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 2, 'max_leaves': None, 'min_child_weight': 1, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 250, 'n_jobs': 1, 'num_parallel_tree': None, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.8, 'sampling_method': None, 'scale_pos_weight': 1.3688085676037482, 'subsample': 1.0, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}`

## Final Test Evaluation
| Metric | Baseline | Improved | Change |
| --- | ---: | ---: | ---: |
| accuracy | 63.73% | 58.08% | -5.65 pp |
| precision | 58.50% | 50.16% | -8.34 pp |
| recall | 47.99% | 83.91% | +35.92 pp |
| f1 | 52.72% | 62.79% | +10.07 pp |
| roc_auc | 67.61% | 68.20% | +0.59 pp |
- Confusion matrix [[TN, FP], [FN, TP]]: [[201, 311], [60, 313]]

## SHAP
- Positive SHAP pushes toward the risk class; negative SHAP pushes away from risk.
| Feature | Mean Absolute SHAP |
| --- | ---: |
| mid_term_score | 0.243824 |
| academic_trend=Declining | 0.222962 |
| academic_trend=Improving | 0.115232 |
| previous_semester_tgpa | 0.072995 |
| ca_average | 0.069002 |
| attendance_percentage | 0.066462 |
| attendance_best2_interaction | 0.063809 |
| ca2_score | 0.058238 |
| attendance_midterm_interaction | 0.057682 |
| best_2_ca_average | 0.041781 |

## Representative Profiles
- LOW_RISK_STUDENT: 0.1131, LOW, MONITOR
- MEDIUM_RISK_STUDENT: 0.4255, MEDIUM, ADVISE
- HIGH_RISK_STUDENT: 0.7242, HIGH, ESCALATE
- SEMESTER_1_STUDENT: 0.4471, MEDIUM, ADVISE
- SENIOR_SEMESTER_STUDENT: 0.6258, MEDIUM, ADVISE

## API
- `POST /api/academic-risk` is preserved.
- Response includes student ID, probability, risk level, SHAP-ranked risk/protective factors, and linked interventions.
