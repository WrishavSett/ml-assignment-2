"""Streamlit app that evaluates the saved breast cancer classifiers on test data.

The test data is either uploaded or taken from the provided test_data.csv. The
five trained pipelines are loaded from the model folder and only used for
prediction; nothing is retrained on the test data.
"""

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
TEST_DATA_PATH = APP_DIR / "test_data.csv"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
DATA_SOURCES = ["Upload a CSV file", "Use the provided test_data.csv"]

COURSE_NAME = "Machine Learning"
COURSE_CODE = "S1-26 AIMLZG565"
STUDENT_NAME = "Wrishav Sett"
STUDENT_ID = "2026AJ05786"

MODEL_DESCRIPTIONS = {
    "Logistic Regression": (
        "A linear model that converts a weighted sum of the scaled features "
        "into a probability of malignancy."
    ),
    "Decision Tree": (
        "A sequence of if-then splits on single features; each leaf predicts "
        "the majority class of the training samples that reach it."
    ),
    "KNN": (
        "Predicts the class of a sample by a vote among its k nearest "
        "training samples in the scaled feature space."
    ),
    "Naive Bayes": (
        "Gaussian Naive Bayes: models each feature as normally distributed "
        "within each class and treats the features as independent given the class."
    ),
    "Support Vector Machine": (
        "Separates the classes with the boundary that has the widest margin; "
        "the kernel lets that boundary curve."
    ),
}
TRAINING_NOTE = (
    "Tuned with GridSearchCV using 5-fold stratified cross-validation on the "
    "455 training samples, scored by MCC, then refitted on the full training set."
)


def read_json(path):
    """Read a JSON file from disk.

    Parameters
    ----------
    path : pathlib.Path
        File to read.

    Returns
    -------
    object
        The parsed JSON content.
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_artifacts():
    """Load the metadata, all five trained models and the provided test data once per server process.

    Returns
    -------
    dict
        feature_names (list of str), target (str), label_map (dict of
        label to 0/1), class_names (list of str), models (dict of
        display name to fitted pipeline, in registry order), best_params
        (dict of display name to tuned hyperparameters) and test_data
        (contents of test_data.csv as bytes).
    """
    registry = read_json(MODEL_DIR / "model_registry.json")
    labels = read_json(MODEL_DIR / "label_map.json")
    return {
        "feature_names": read_json(MODEL_DIR / "feature_names.json"),
        "target": labels["target"],
        "label_map": labels["label_map"],
        "class_names": labels["class_names"],
        "models": {name: joblib.load(MODEL_DIR / file_name) for name, file_name in registry["models"].items()},
        "best_params": read_json(MODEL_DIR / "best_params.json"),
        "test_data": TEST_DATA_PATH.read_bytes(),
    }


def read_upload(uploaded_file):
    """Read an uploaded file as a CSV table.

    Parameters
    ----------
    uploaded_file : file-like object
        The file returned by the Streamlit file uploader, or the provided
        test data wrapped in a bytes buffer.

    Returns
    -------
    pandas.DataFrame
        The uploaded table.

    Raises
    ------
    ValueError
        If the file cannot be parsed as CSV or has no rows.
    """
    try:
        frame = pd.read_csv(uploaded_file)
    except Exception as error:
        raise ValueError(f"Could not read the file as CSV ({error}).") from error
    if frame.empty:
        raise ValueError("The uploaded file has no data rows.")
    return frame


def prepare_data(frame, feature_names, target, label_map):
    """Check the uploaded table and split it into model inputs and encoded labels.

    Parameters
    ----------
    frame : pandas.DataFrame
        The uploaded table.
    feature_names : list of str
        Feature columns in the order the models were trained on.
    target : str
        Name of the target column.
    label_map : dict
        Mapping from target labels to 0/1, with malignant as 1.

    Returns
    -------
    X : pandas.DataFrame
        Feature columns in training order; extra columns are dropped.
    y : numpy.ndarray
        Encoded labels.

    Raises
    ------
    ValueError
        If columns are missing, labels are unknown, or feature values are
        empty or non-numeric.
    """
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}.")
    if target not in frame.columns:
        raise ValueError(f"Missing target column: {target}.")

    if frame[target].isna().any():
        raise ValueError(f"Empty values in the {target} column.")
    labels = frame[target].astype(str).str.strip()
    unknown = sorted(set(labels) - set(label_map))
    if unknown:
        raise ValueError(
            f"Unexpected values in {target}: {', '.join(unknown)}. "
            f"Expected only {' or '.join(label_map)}."
        )

    X = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    invalid = [name for name in feature_names if X[name].isna().any()]
    if invalid:
        raise ValueError(f"Empty or non-numeric values in: {', '.join(invalid)}.")

    return X, labels.map(label_map).to_numpy(dtype=int)


def positive_class_scores(model, X):
    """Return a continuous score for the malignant class, used for AUC.

    Parameters
    ----------
    model : fitted sklearn estimator
        Classifier to score.
    X : pandas.DataFrame
        Features in training order.

    Returns
    -------
    numpy.ndarray
        Predicted probability of malignancy when the model provides
        predict_proba, otherwise the decision function (used by the SVM).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_model(model, X, y):
    """Compute the six evaluation metrics, using the same definitions as the notebook.

    Parameters
    ----------
    model : fitted sklearn estimator
        Classifier to evaluate.
    X : pandas.DataFrame
        Features in training order.
    y : numpy.ndarray
        True labels, 1 = malignant and 0 = benign.

    Returns
    -------
    metrics : dict
        Accuracy, AUC, Precision, Recall, F1 and MCC. AUC is None when the
        labels contain only one class, because it is undefined there.
    y_pred : numpy.ndarray
        Predicted labels.
    """
    y_pred = model.predict(X)
    auc = roc_auc_score(y, positive_class_scores(model, X)) if len(np.unique(y)) == 2 else None
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


def confusion_matrix_figure(y, y_pred, class_names):
    """Draw the confusion matrix for one model's predictions.

    Parameters
    ----------
    y : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.
    class_names : list of str
        Display names for labels 0 and 1.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with rows as actual classes and columns as predictions.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, labels=[0, 1], display_labels=class_names,
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    return fig


def classification_report_table(y, y_pred, class_names):
    """Build the classification report as a table, laid out as sklearn prints it.

    Parameters
    ----------
    y : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.
    class_names : list of str
        Display names for labels 0 and 1.

    Returns
    -------
    pandas.io.formats.style.Styler
        Precision, recall, F1-score and support for each class, followed by
        the accuracy, macro average and weighted average rows.
    """
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=class_names, output_dict=True, zero_division=0
    )
    # Accuracy is a single value, so as in the printed report it fills only the F1-score and support cells
    report["accuracy"] = {"f1-score": report["accuracy"], "support": len(y)}
    rows = class_names + ["accuracy", "macro avg", "weighted avg"]
    table = pd.DataFrame(report).T.loc[rows, ["precision", "recall", "f1-score", "support"]]
    table.columns = ["Precision", "Recall", "F1-score", "Support"]
    return table.style.format("{:.4f}", na_rep="").format("{:.0f}", subset=["Support"])


def show_model_card(model_name, model, best_params):
    """Show what the selected model is, how it was trained and its tuned hyperparameters.

    Parameters
    ----------
    model_name : str
        Display name of the model.
    model : sklearn.pipeline.Pipeline
        The fitted pipeline.
    best_params : dict
        Tuned hyperparameter names and values for this model.
    """
    steps = ", then ".join(type(step).__name__ for _, step in model.steps)
    with st.expander(f"Model card: {model_name}"):
        st.write(MODEL_DESCRIPTIONS[model_name])
        st.markdown(f"**Pipeline:** {steps}\n\n**Training:** {TRAINING_NOTE}\n\n**Tuned hyperparameters**")
        st.table(
            pd.DataFrame({"Hyperparameter": list(best_params), "Value": [str(value) for value in best_params.values()]}),
            hide_index=True,
        )


def show_metrics(metrics):
    """Show the six metrics as a row of tiles.

    Parameters
    ----------
    metrics : dict
        Output of evaluate_model.
    """
    for column, name in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
        value = metrics[name]
        column.metric(name, "n/a" if value is None else f"{value:.4f}")
    if metrics["AUC"] is None:
        st.caption("AUC needs both classes in the uploaded data.")


def main():
    """Build the page: test data, model selection, model card, metrics, confusion matrix and classification report."""
    st.set_page_config(page_title="Breast Cancer Model Evaluation", layout="wide")
    artifacts = load_artifacts()

    st.title("Breast Cancer Diagnosis: Model Evaluation")
    st.write(
        "Evaluate five classifiers trained on the Breast Cancer Wisconsin (Diagnostic) "
        "dataset. Upload the test data or use the provided test set, pick a model, and "
        "the saved model scores it without retraining. Malignant is the positive class."
    )

    with st.sidebar:
        st.header("1. Test data")
        source = st.radio("Data source", DATA_SOURCES)
        if source == DATA_SOURCES[0]:
            data_file = st.file_uploader("Upload test_data.csv", type=["csv"])
        else:
            data_file = io.BytesIO(artifacts["test_data"])
        # "ignore" downloads the file without rerunning the app
        st.download_button(
            "Download test_data.csv", artifacts["test_data"],
            file_name="test_data.csv", mime="text/csv", on_click="ignore",
        )
        st.header("2. Model")
        model_name = st.selectbox("Select a model", list(artifacts["models"]))
        st.divider()
        st.header(COURSE_NAME)
        st.markdown(f"{COURSE_CODE}\n\n{STUDENT_NAME}\n\n{STUDENT_ID}")

    model = artifacts["models"][model_name]
    show_model_card(model_name, model, artifacts["best_params"][model_name])

    if data_file is None:
        st.info(
            f"Upload a CSV with the {len(artifacts['feature_names'])} feature columns "
            f"and the {artifacts['target']} column (M or B), or choose the provided "
            "test data in the sidebar, to see the results."
        )
        st.stop()

    try:
        frame = read_upload(data_file)
        X, y = prepare_data(frame, artifacts["feature_names"], artifacts["target"], artifacts["label_map"])
    except ValueError as error:
        st.error(str(error))
        st.stop()

    metrics, y_pred = evaluate_model(model, X, y)

    st.subheader(f"{model_name}: evaluation metrics")
    show_metrics(metrics)

    # Side by side on wide screens; Streamlit stacks the columns on narrow ones
    matrix_column, report_column = st.columns(2)
    matrix_column.subheader("Confusion matrix")
    fig = confusion_matrix_figure(y, y_pred, artifacts["class_names"])
    matrix_column.pyplot(fig)
    plt.close(fig)
    report_column.subheader("Classification report")
    report_column.table(classification_report_table(y, y_pred, artifacts["class_names"]))


if __name__ == "__main__":
    main()