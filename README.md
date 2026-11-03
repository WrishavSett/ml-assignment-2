# Breast Cancer Diagnosis: Comparing Five Classification Models

### Machine Learning

S1-26 AIMLZG565

Wrishav Sett

2026AJ05786

Live app: https://2026aj05786-ml-assignment-2.streamlit.app/

## Problem statement

Predict whether a breast mass is malignant or benign from 30 measurements of cell nuclei taken from a fine needle aspirate image. Five classification models are trained and compared on the same data, and the trained models are deployed in a Streamlit app that evaluates them on held-out test data.

## Dataset description

**Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository, dataset id 17: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

- 569 instances and 30 numeric features, with no missing values and no duplicate rows.
- The features are computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. Ten characteristics of the cell nuclei are measured, and each is recorded three ways: the mean across nuclei (`_mean`), its standard error (`_se`) and the mean of the three largest values (`_worst`). Columns are named accordingly, for example `radius_mean`, `radius_se` and `radius_worst`.
- Target: `diagnosis`, malignant (M, 212 samples, 37.26%) or benign (B, 357 samples, 62.74%). Malignant is the positive class (M = 1, B = 0).
- Split: stratified 80/20 with random seed 786, giving 455 training samples (170 M, 285 B) and 114 test samples (42 M, 72 B). The test samples are saved as `test_data.csv` with all 30 features and the `diagnosis` column.

| Characteristic | What it measures |
|---|---|
| radius | Mean distance from the centre to points on the perimeter |
| texture | Standard deviation of grey-scale values |
| perimeter | Perimeter of the nucleus |
| area | Area of the nucleus |
| smoothness | Local variation in radius lengths |
| compactness | perimeter^2 / area - 1.0 |
| concavity | Severity of concave portions of the contour |
| concave_points | Number of concave portions of the contour |
| symmetry | Symmetry of the nucleus |
| fractal_dimension | Coastline approximation - 1 |

## GitHub Repository Link and Structure

https://github.com/WrishavSett/ml-assignment-2

```
ml-assignment-2/
|-- .streamlit/
|   |-- config.toml
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- feature_names.json
|   |-- label_map.json
|   |-- model_registry.json
|   |-- best_params.json
|-- 2026AJ05786.ipynb
|-- README.md
|-- app.py
|-- requirements.txt
|-- test_data.csv
```

## Models used

### How the models were trained

Each model is a scikit-learn pipeline of a `StandardScaler` followed by the classifier, so the scaler is fitted on training data only and is saved together with the model. Naive Bayes is Gaussian Naive Bayes. Hyperparameters were tuned with `GridSearchCV` using 5-fold stratified cross-validation on the training set, scored by MCC. All five models were tuned the same way, so the comparison reflects the algorithms rather than their default settings. Each best model was refitted on the full training set, evaluated once on the 114-sample test set, and saved with joblib. The Streamlit app loads these saved models and does not retrain them.

### Tuned hyperparameters

| Model | Tuned parameters |
|---|---|
| Logistic Regression | C=100, class_weight=None |
| Decision Tree | class_weight=None, criterion=gini, max_depth=4, min_samples_leaf=2 |
| KNN | n_neighbors=15, p=2, weights=distance |
| Naive Bayes | var_smoothing=1e-12 |
| Support Vector Machine | C=10, class_weight=None, gamma=0.01, kernel=rbf |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9957 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9474 | 0.9312 | 0.9737 | 0.8810 | 0.9250 | 0.8872 |
| KNN | 0.9561 | 0.9964 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9924 | 0.9487 | 0.8810 | 0.9136 | 0.8675 |
| Support Vector Machine | 0.9912 | 1.0000 | 0.9767 | 1.0000 | 0.9882 | 0.9814 |

Metrics are computed on the 114-sample test set (`test_data.csv`) with malignant as the positive class. AUC uses the predicted probability of malignancy, except for the SVM, which uses its decision function.

### Observations

Section numbers refer to `2026AJ05786.ipynb`.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best on test MCC (0.9623), with accuracy 0.9825 and AUC 0.9957: 1 false negative and 1 false positive out of 114 test samples. The classes are almost linearly separable (Section 2.3), so a single linear boundary captures most of the signal, and tuning preferred weak regularisation (C = 100). Its cross-validated MCC (0.9395 +/- 0.0278) varies less from fold to fold than that of the SVM or KNN, so it is the most predictable of the top three models. The trade-off is interpretability: radius, perimeter and area are near duplicates of each other (Section 2.1), so the model has to split their combined effect across several correlated coefficients. Individual coefficients therefore cannot be read as the effect of one feature, even though the combined prediction is accurate. |
| Decision Tree | Fourth on test MCC (0.8872) and the lowest AUC of all models (0.9312). Tuning chose a shallow tree (max_depth = 4, min_samples_leaf = 2); a tree of depth 4 has at most 16 leaves, so it can output only a few distinct probabilities, and this coarse ranking produces its step-like ROC curve (Section 8) and low AUC. It missed 5 of the 42 malignant test samples (recall 0.8810) while raising 1 false alarm. Training accuracy (0.9802) is well above its cross-validated (0.9385) and test (0.9474) accuracy, and its cross-validated precision varies the most of all models (+/- 0.0590), both signs of high variance. Its tuning curve (Section 5.1) is almost flat across depths, so the limit is the axis-aligned splits themselves: the class boundary is oblique, and a shallow tree can only approximate it with a few steps. |
| KNN | Third on test MCC (0.9054): 3 false negatives and 2 false positives. Tuning chose a large neighbourhood (k = 15) with distance weighting and Euclidean distance. Its ranking of samples is excellent (AUC 0.9964), but its hard predictions lean towards benign: it has the highest cross-validated precision of all models (0.9943) and the most variable cross-validated recall (0.9000 +/- 0.0634). Two of its three missed malignant samples are also missed by the Decision Tree and Naive Bayes (Section 9.1), which places them in the overlap region between the classes, where benign neighbours, the majority class, can outvote them. Scaling puts all features on the same range, but the redundant size features still count several times in the distance, which dilutes the contribution of the other characteristics. |
| Naive Bayes | Lowest test MCC (0.8675) and accuracy (0.9386): 5 false negatives and 2 false positives. Its assumptions fit this dataset poorly: the features are strongly correlated (15 pairs above 0.95) rather than independent, and several are right-skewed rather than Gaussian. Treating correlated features as independent evidence counts the same size information several times, which distorts its probability estimates. Its training accuracy (0.9429) is barely above its test accuracy (Section 9.1), so its errors come from bias in its assumptions rather than from overfitting, and tuning var_smoothing made no difference to its score (Section 5.1). Its AUC (0.9924) is still high, so it ranks samples well; it is fast and simple, but the decisions it makes at the default threshold are the least accurate of the five models. |
| Support Vector Machine | Best on all six test metrics: accuracy 0.9912, AUC 1.0000, recall 1.0000 and MCC 0.9814, with no missed malignant samples and 1 false positive. Tuning chose an RBF kernel with C = 10 and gamma = 0.01, lower than the default gamma (about 0.033 for 30 standardised features), which gives a smooth, gently curved boundary; this suits classes that are almost linearly separable (Section 2.3), and the linear kernel was close behind in the tuning curves (Section 5.1). It is also the only model that correctly classifies the malignant test sample missed by the other four. Its test MCC is above its cross-validated MCC (0.9535 +/- 0.0389), so the perfect test recall partly reflects a favourable split; its cross-validated recall of 0.9471 is a more realistic expectation for new data. |
| Overall, Winner for your dataset? | Support Vector Machine. It has the highest test MCC (0.9814), which is the selection rule, and also the highest cross-validated MCC (0.9535), so both criteria agree. The ranking by test MCC (Support Vector Machine, Logistic Regression, KNN, Decision Tree, Naive Bayes) is the same as the ranking by cross-validated MCC, so the order is not an artefact of one test split. Logistic Regression is a close runner-up: it made only one more error on the test set, and the gap in cross-validated MCC (0.0140) is smaller than either model's fold-to-fold spread. It would be a reasonable choice where a simpler linear model is preferred. |