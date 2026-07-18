import io
import pickle
from typing import Dict, List, Optional, Tuple

import cloudpickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, r2_score,
                             roc_auc_score)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """A dynamic preprocessing wrapper for dataset cleaning, encoding, and scaling."""

    def __init__(self):
        self.target_column: Optional[str] = None
        self.raw_columns: List[str] = []
        self.feature_names: List[str] = []
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.one_hot_columns: Dict[str, List[str]] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.missing_fill_values: Dict[str, object] = {}
        self.scaler: Optional[StandardScaler] = None
        self.removed_constant_columns: List[str] = []
        self.removed_outliers: int = 0
        self.duplicate_count: int = 0
        self.outlier_bounds: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        self.target_column = target_column

        if target_column in df.columns:
            df = df.drop(columns=[target_column])

        self.raw_columns = df.columns.tolist()
        self.numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        self.duplicate_count = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)

        for col in self.numerical_columns:
            median_value = df[col].median() if not df[col].dropna().empty else 0.0
            self.missing_fill_values[col] = median_value
            df[col] = df[col].fillna(median_value)

        for col in self.categorical_columns:
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "missing"
            self.missing_fill_values[col] = mode_value
            df[col] = df[col].fillna(mode_value).astype(str)

        self.removed_constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if self.removed_constant_columns:
            df = df.drop(columns=self.removed_constant_columns)

        self.label_encoders = {}
        self.one_hot_columns = {}

        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            unique_count = df[col].nunique()
            if unique_count <= 2:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
                self.one_hot_columns[col] = dummies.columns.tolist()
                df = df.drop(columns=[col]).join(dummies)

        self.outlier_bounds = {}
        outlier_mask = pd.Series(False, index=df.index)
        for col in self.numerical_columns:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.outlier_bounds[col] = {"lower": lower, "upper": upper}
            outlier_mask |= ~df[col].between(lower, upper)

        self.removed_outliers = int(outlier_mask.sum())
        if self.removed_outliers > 0:
            df = df.loc[~outlier_mask].reset_index(drop=True)

        self.scaler = StandardScaler()
        numeric_columns_to_scale = [col for col in self.numerical_columns if col in df.columns]
        if numeric_columns_to_scale:
            df[numeric_columns_to_scale] = self.scaler.fit_transform(df[numeric_columns_to_scale])

        self.feature_names = df.columns.tolist()
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])

        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(self.missing_fill_values.get(col, 0.0))

        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(self.missing_fill_values.get(col, "missing")).astype(str)

        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            if col in self.label_encoders:
                encoder = self.label_encoders[col]
                df[col] = encoder.transform(df[col].astype(str))
            else:
                dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
                for dummy_col in self.one_hot_columns.get(col, []):
                    df[dummy_col] = dummies.get(dummy_col, 0)
                df = df.drop(columns=[col])

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df.reindex(columns=self.feature_names, fill_value=0)

        numeric_columns_to_scale = [col for col in self.numerical_columns if col in df.columns]
        if numeric_columns_to_scale and self.scaler is not None:
            df[numeric_columns_to_scale] = self.scaler.transform(df[numeric_columns_to_scale])

        return df


@st.cache_data
def load_dataset(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return df


def detect_task_type(y: pd.Series) -> str:
    if y.dtype == object or y.dtype.name == "category" or y.nunique() <= 10:
        return "Classification"
    return "Regression"


def summarize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    dtypes = pd.DataFrame({"Column": df.columns, "Data Type": [str(dt) for dt in df.dtypes]})
    summary = df.describe(include="all").transpose()
    return missing, dtypes, summary


def classify_models(task_type: str):
    if task_type == "Classification":
        return [
            ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=100)),
            ("Decision Tree", DecisionTreeClassifier(random_state=42)),
            ("Logistic Regression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ]
    return [
        ("Random Forest", RandomForestRegressor(random_state=42, n_estimators=100)),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Linear Regression", LinearRegression()),
    ]


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        if y_proba is not None:
            if len(np.unique(y_true)) == 2:
                metrics["ROC AUC"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["ROC AUC"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        else:
            metrics["ROC AUC"] = np.nan
    except Exception:
        metrics["ROC AUC"] = np.nan
    return metrics


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_training_report(models, X_train, X_test, y_train, y_test, X_full, y_full, task_type: str):
    results = []
    progress = st.progress(0)
    for idx, (name, model) in enumerate(models):
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        if task_type == "Classification":
            y_proba_test = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            y_proba_train = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
            metrics = compute_classification_metrics(y_test, y_pred_test, y_proba_test)
            train_score = float(accuracy_score(y_train, y_pred_train))
            test_score = float(accuracy_score(y_test, y_pred_test))
            cv_values = cross_val_score(model, X_full, y_full, cv=5, scoring="accuracy")
        else:
            y_proba_test = None
            metrics = compute_regression_metrics(y_test, y_pred_test)
            train_score = float(r2_score(y_train, y_pred_train))
            test_score = float(r2_score(y_test, y_pred_test))
            cv_values = cross_val_score(model, X_full, y_full, cv=5, scoring="r2")

        results.append({
            "Model": name,
            "Model Object": model,
            "Train Score": train_score,
            "Test Score": test_score,
            "CV Score": float(np.mean(cv_values)),
            **metrics,
            "Confusion Matrix": confusion_matrix(y_test, y_pred_test) if task_type == "Classification" else None,
        })
        progress.progress(int((idx + 1) / len(models) * 100))
    progress.empty()
    return results


def build_comparison_table(results: List[Dict]) -> pd.DataFrame:
    table = pd.DataFrame([
        {
            "Model": result["Model"],
            "Train Score": round(result["Train Score"], 4),
            "Test Score": round(result["Test Score"], 4),
            "CV Score": round(result["CV Score"], 4),
            **{k: round(v, 4) for k, v in result.items() if k not in ["Model", "Model Object", "Train Score", "Test Score", "CV Score", "Confusion Matrix"]},
        }
        for result in results
    ])
    return table


def create_radar_chart(results: List[Dict], task_type: str):
    if not results:
        return None

    categories = []
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"] if task_type == "Classification" else ["R2", "MAE", "MSE", "RMSE"]:
        categories.append(metric)

    fig = go.Figure()
    for result in results:
        values = []
        for metric in categories:
            value = result.get(metric, np.nan)
            if task_type == "Regression" and metric in ["MAE", "MSE", "RMSE"]:
                values.append(1.0 / (1.0 + value) if np.isfinite(value) else 0.0)
            else:
                values.append(value if np.isfinite(value) else 0.0)
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=result["Model"],
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title_text="Model Performance Radar"
    )
    return fig


def get_feature_importance(result, feature_names: List[str]) -> Optional[pd.DataFrame]:
    model = result.get("Model Object")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return importance_df
    return None


def build_pdf_report(
    raw_df: pd.DataFrame,
    task_type: str,
    target_column: str,
    comparison_table: pd.DataFrame,
    best_model_name: str,
    preprocessing_report: Dict[str, object],
) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Advanced Machine Learning Training Studio Report", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Task Type: {task_type}")
    pdf.multi_cell(0, 8, f"Target Column: {target_column}")
    pdf.multi_cell(0, 8, f"Dataset Shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Preprocessing Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    for key, value in preprocessing_report.items():
        pdf.multi_cell(0, 8, f"- {key}: {value}")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Model Comparison", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Best Model: {best_model_name}")
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    for _, row in comparison_table.iterrows():
        pdf.multi_cell(0, 8, f"{row['Model']}: Train {row['Train Score']}, Test {row['Test Score']}, CV {row['CV Score']}")

    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes


def create_prediction_inputs(raw_df: pd.DataFrame, target_column: str) -> Dict[str, object]:
    inputs = {}
    feature_columns = [col for col in raw_df.columns if col != target_column]
    for col in feature_columns:
        if raw_df[col].dtype == object or raw_df[col].dtype.name == "category":
            options = sorted(raw_df[col].dropna().unique().astype(str).tolist())
            inputs[col] = st.sidebar.selectbox(f"{col}", options, key=f"input_{col}")
        elif raw_df[col].dtype == bool:
            options = [True, False]
            inputs[col] = st.sidebar.selectbox(f"{col}", options, key=f"input_{col}")
        else:
            feature_min = float(raw_df[col].min()) if pd.notna(raw_df[col].min()) else 0.0
            feature_max = float(raw_df[col].max()) if pd.notna(raw_df[col].max()) else feature_min + 1.0
            feature_mean = float(raw_df[col].mean()) if pd.notna(raw_df[col].mean()) else 0.0
            inputs[col] = st.sidebar.number_input(
                f"{col}", min_value=feature_min, max_value=feature_max, value=feature_mean, format="%f", key=f"input_{col}"
            )
    return inputs


def build_model_artifact(preprocessor: DataPreprocessor, model, target_column: str) -> bytes:
    artifact = {
        "preprocessor": preprocessor,
        "model": model,
        "target_column": target_column,
    }
    try:
        return pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    except pickle.PicklingError:
        return cloudpickle.dumps(artifact)


def get_prediction_confidence(task_type: str, model, X_sample: pd.DataFrame, y_train: pd.Series, prediction):
    if task_type == "Classification" and hasattr(model, "predict_proba"):
        return None
    if task_type == "Regression":
        std = float(np.std(y_train)) if len(y_train) > 1 else 0.0
        if std == 0:
            return 1.0
        confidence = max(0.0, min(1.0, 1.0 - abs(prediction - float(np.mean(y_train))) / std))
        return round(confidence, 4)
    return None


def main():
    st.set_page_config(page_title="Advanced ML Training Studio", layout="wide", initial_sidebar_state="expanded")
    st.title("Advanced Machine Learning Training Studio")
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Choose a section",
        [
            "Data Upload",
            "Preprocessing",
            "EDA Dashboard",
            "Model Training",
            "Model Comparison",
            "Feature Importance",
            "Prediction Center",
            "Reports",
            "Model Management",
        ],
    )

    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    if "df_preprocessed" not in st.session_state:
        st.session_state.df_preprocessed = None
    if "task_type" not in st.session_state:
        st.session_state.task_type = None
    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    if "model_results" not in st.session_state:
        st.session_state.model_results = []
    if "best_model_name" not in st.session_state:
        st.session_state.best_model_name = None
    if "best_model" not in st.session_state:
        st.session_state.best_model = None
    if "metrics_table" not in st.session_state:
        st.session_state.metrics_table = None
    if "preprocessing_report" not in st.session_state:
        st.session_state.preprocessing_report = None
    if "model_artifact" not in st.session_state:
        st.session_state.model_artifact = None
    if "training_dataset" not in st.session_state:
        st.session_state.training_dataset = None

    if page == "Data Upload":
        st.header("1. Upload your dataset")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Upload any CSV dataset for automatic ML training.")
        if uploaded_file is not None:
            try:
                df = load_dataset(uploaded_file)
                st.session_state.df_raw = df
                st.success("Dataset loaded successfully.")
            except Exception as exc:
                st.error(f"Failed to load dataset: {exc}")

        if st.session_state.df_raw is not None:
            df = st.session_state.df_raw
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", int(df.isna().sum().sum()))

            with st.expander("Dataset Structure"):
                st.write(df.dtypes.astype(str).to_frame("Data Type"))
            with st.expander("Missing Value Summary"):
                missing, dtypes, summary = summarize_dataframe(df)
                st.write(missing)
            with st.expander("Statistical Summary"):
                st.write(summary)

    elif page == "Preprocessing":
        st.header("2. Data Preprocessing")
        if st.session_state.df_raw is None:
            st.warning("Upload a dataset first in the Data Upload section.")
        else:
            if st.button("Run Preprocessing") or st.session_state.df_preprocessed is None:
                preprocessor = DataPreprocessor()
                processed_df = preprocessor.fit(st.session_state.df_raw)
                st.session_state.preprocessor = preprocessor
                st.session_state.df_preprocessed = processed_df
                st.session_state.preprocessing_report = {
                    "Original Rows": st.session_state.df_raw.shape[0],
                    "Original Columns": st.session_state.df_raw.shape[1],
                    "Duplicate Rows Removed": preprocessor.duplicate_count,
                    "Columns Removed (Constant)": len(preprocessor.removed_constant_columns),
                    "Rows Removed (Outliers)": preprocessor.removed_outliers,
                    "Features After Encoding": len(preprocessor.feature_names),
                    "Numeric Features Scaled": len(preprocessor.numerical_columns),
                    "Categorical Features Encoded": len(preprocessor.categorical_columns),
                }

            st.subheader("Preprocessing Report")
            st.write(st.session_state.preprocessing_report)
            with st.expander("Preprocessing Details"):
                st.write(f"Removed constant columns: {preprocessor.removed_constant_columns}")
                st.write("Outlier thresholds:")
                st.write(preprocessor.outlier_bounds)
                st.write("Encoded categorical mappings:")
                st.write({
                    key: "Label Encoding" if key in preprocessor.label_encoders else values
                    for key, values in preprocessor.one_hot_columns.items()
                })

            st.subheader("Processed Data Preview")
            st.dataframe(st.session_state.df_preprocessed.head(10), use_container_width=True)
            st.write(f"Processed shape: {st.session_state.df_preprocessed.shape}")

    elif page == "EDA Dashboard":
        st.header("3. Exploratory Data Analysis")
        if st.session_state.df_raw is None:
            st.warning("Upload a dataset first in the Data Upload section.")
        else:
            df = st.session_state.df_raw.copy()
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            st.subheader("Interactive Charts")
            tab1, tab2, tab3 = st.tabs(["Histograms", "Scatter & Box", "Correlation + Pairplot"])

            with tab1:
                hist_col = st.selectbox("Histogram Column", numeric_cols + categorical_cols, index=0)
                fig = px.histogram(df, x=hist_col, nbins=40, title=f"Distribution of {hist_col}")
                st.plotly_chart(fig, use_container_width=True)

                dist_col = st.selectbox("Distribution Column", numeric_cols, index=0)
                fig2 = px.histogram(df, x=dist_col, marginal="box", title=f"Distribution and Boxplot for {dist_col}")
                st.plotly_chart(fig2, use_container_width=True)

            with tab2:
                x_axis = st.selectbox("X axis", numeric_cols, index=0)
                y_axis = st.selectbox("Y axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                color_col = st.selectbox("Color by", [None] + categorical_cols, index=0)
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title=f"Scatterplot of {y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)

                box_col = st.selectbox("Boxplot Column", numeric_cols, index=0, key="boxplot_col")
                box_group = st.selectbox("Group by", [None] + categorical_cols, index=0, key="boxplot_group")
                fig_box = px.box(df, x=box_group if box_group else None, y=box_col, title=f"Boxplot for {box_col}")
                st.plotly_chart(fig_box, use_container_width=True)

            with tab3:
                if numeric_cols:
                    fig_heatmap = px.imshow(df[numeric_cols].corr(), text_auto=True, title="Correlation Heatmap")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    pair_columns = st.multiselect("Pairplot Columns", numeric_cols, default=numeric_cols[:4])
                    if len(pair_columns) >= 2:
                        fig_pair = px.scatter_matrix(df[pair_columns], dimensions=pair_columns, title="Pairplot")
                        st.plotly_chart(fig_pair, use_container_width=True)
                    else:
                        st.info("Select at least two numeric features for a pairplot.")
                else:
                    st.info("No numeric columns available for correlation or pairplot.")

            if st.session_state.target_column:
                st.subheader("Target Analysis")
                target = st.session_state.target_column
                if target in df.columns:
                    if df[target].dtype == object or df[target].dtype.name == "category":
                        fig = px.histogram(df, x=target, title=f"Target distribution for {target}")
                    else:
                        fig = px.histogram(df, x=target, nbins=30, title=f"Target distribution for {target}")
                    st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Training":
        st.header("4. Model Training")
        if st.session_state.df_raw is None or st.session_state.df_preprocessed is None:
            st.warning("Upload and preprocess a dataset first.")
        else:
            df = st.session_state.df_raw.copy()
            target_column = st.selectbox("Select target column", df.columns, index=len(df.columns) - 1)
            st.session_state.target_column = target_column
            y = df[target_column]
            task_type = detect_task_type(y)
            st.session_state.task_type = task_type
            st.markdown(f"**Detected Task**: {task_type}")

            test_size = st.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
            if st.button("Train Models"):
                preprocessor = DataPreprocessor()
                preprocessor.fit(st.session_state.df_raw, target_column=target_column)
                st.session_state.preprocessor = preprocessor
                st.session_state.df_preprocessed = preprocessor.transform(st.session_state.df_raw)
                st.session_state.preprocessing_report = {
                    "Original Rows": st.session_state.df_raw.shape[0],
                    "Original Columns": st.session_state.df_raw.shape[1],
                    "Duplicate Rows Removed": preprocessor.duplicate_count,
                    "Columns Removed (Constant)": len(preprocessor.removed_constant_columns),
                    "Rows Removed (Outliers)": preprocessor.removed_outliers,
                    "Features After Encoding": len(preprocessor.feature_names),
                    "Numeric Features Scaled": len(preprocessor.numerical_columns),
                    "Categorical Features Encoded": len(preprocessor.categorical_columns),
                }
                X = st.session_state.df_preprocessed
                y = st.session_state.df_raw[target_column]
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y if task_type == "Classification" else None,
                )
                st.session_state.training_dataset = (X_train, X_test, y_train, y_test, X, y)
                models = classify_models(task_type)
                results = build_training_report(models, X_train, X_test, y_train, y_test, X, y, task_type)
                st.session_state.model_results = results
                results_df = build_comparison_table(results)
                best = sorted(results, key=lambda x: x["CV Score"], reverse=True)[0]
                st.session_state.best_model_name = best["Model"]
                st.session_state.best_model = best["Model Object"]
                st.session_state.metrics_table = results_df
                st.session_state.model_artifact = build_model_artifact(st.session_state.preprocessor, best["Model Object"], target_column)

            if st.session_state.model_results:
                st.subheader("Training Results")
                for result in st.session_state.model_results:
                    with st.expander(result["Model"]):
                        st.metric("Train Score", round(result["Train Score"], 4))
                        st.metric("Test Score", round(result["Test Score"], 4))
                        st.metric("CV Score", round(result["CV Score"], 4))
                        if task_type == "Classification":
                            st.write({k: round(v, 4) for k, v in result.items() if k in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]})
                            cm = result["Confusion Matrix"]
                            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title=f"Confusion Matrix ({result['Model']})")
                            st.plotly_chart(fig_cm, use_container_width=True)
                        else:
                            st.write({k: round(v, 4) for k, v in result.items() if k in ["MAE", "MSE", "RMSE", "R2"]})

    elif page == "Model Comparison":
        st.header("5. Model Comparison")
        if not st.session_state.model_results:
            st.warning("Train models first to compare results.")
        else:
            table = st.session_state.metrics_table
            st.dataframe(table, use_container_width=True)
            st.subheader("Ranking")
            ranking = table.sort_values(by="CV Score", ascending=False).reset_index(drop=True)
            ranking.index += 1
            st.dataframe(ranking, use_container_width=True)

            fig_bar = px.bar(ranking, x="Model", y=["Train Score", "Test Score", "CV Score"], barmode="group", title="Model Score Comparison")
            st.plotly_chart(fig_bar, use_container_width=True)

            radar = create_radar_chart(st.session_state.model_results, st.session_state.task_type)
            if radar is not None:
                st.plotly_chart(radar, use_container_width=True)

            st.success(f"Best model identified: {st.session_state.best_model_name}")

    elif page == "Feature Importance":
        st.header("6. Feature Importance")
        if not st.session_state.model_results:
            st.warning("Train models first to show feature importance.")
        else:
            for result in st.session_state.model_results:
                importance_df = get_feature_importance(result, st.session_state.preprocessor.feature_names)
                if importance_df is not None:
                    st.subheader(f"{result['Model']} Feature Importance")
                    top_n = importance_df.head(10)
                    st.dataframe(top_n, use_container_width=True)
                    fig = px.bar(top_n, x="Importance", y="Feature", orientation="h", title=f"Top Features for {result['Model']}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(f"{result['Model']} does not support tree-based feature importance.")

    elif page == "Prediction Center":
        st.header("7. Prediction Center")
        if st.session_state.df_raw is None:
            st.warning("Upload a dataset and train a model first.")
        elif not st.session_state.model_results or st.session_state.best_model is None:
            st.warning("Train a model first in Model Training.")
        else:
            target_column = st.session_state.target_column
            st.subheader("Enter feature values for prediction")
            input_features = [col for col in st.session_state.df_raw.columns if col != target_column]
            user_inputs = {}
            cols = st.columns(2)
            for count, col in enumerate(input_features):
                if st.session_state.df_raw[col].dtype == object or st.session_state.df_raw[col].dtype.name == "category":
                    options = sorted(st.session_state.df_raw[col].dropna().unique().astype(str).tolist())
                    user_inputs[col] = cols[count % 2].selectbox(col, options, key=f"pred_{col}")
                elif st.session_state.df_raw[col].dtype == bool:
                    user_inputs[col] = cols[count % 2].selectbox(col, [True, False], key=f"pred_{col}")
                else:
                    feature_min = float(st.session_state.df_raw[col].min()) if pd.notna(st.session_state.df_raw[col].min()) else 0.0
                    feature_max = float(st.session_state.df_raw[col].max()) if pd.notna(st.session_state.df_raw[col].max()) else feature_min + 1.0
                    feature_mean = float(st.session_state.df_raw[col].mean()) if pd.notna(st.session_state.df_raw[col].mean()) else 0.0
                    user_inputs[col] = cols[count % 2].number_input(
                        col,
                        min_value=feature_min,
                        max_value=feature_max,
                        value=feature_mean,
                        format="%f",
                        key=f"pred_{col}",
                    )

            if st.button("Generate Prediction"):
                input_df = pd.DataFrame([user_inputs])
                processed_input = st.session_state.preprocessor.transform(input_df)
                processed_input = processed_input.reindex(columns=st.session_state.preprocessor.feature_names, fill_value=0)
                selected_model = st.session_state.best_model
                prediction = selected_model.predict(processed_input)
                st.subheader("Prediction Result")
                if st.session_state.task_type == "Classification":
                    st.write(f"Predicted class: {prediction[0]}")
                    if hasattr(selected_model, "predict_proba"):
                        proba = selected_model.predict_proba(processed_input)[0]
                        labels = selected_model.classes_
                        proba_df = pd.DataFrame({"Class": labels, "Probability": proba})
                        st.dataframe(proba_df, use_container_width=True)
                        confidence = float(np.max(proba))
                        st.metric("Confidence Score", round(confidence, 4))
                    else:
                        st.info("Probability not available for this model.")
                else:
                    value = float(prediction[0])
                    st.write(f"Predicted value: {round(value, 4)}")
                    confidence = get_prediction_confidence(
                        st.session_state.task_type,
                        selected_model,
                        processed_input,
                        st.session_state.training_dataset[2],
                        value,
                    )
                    if confidence is not None:
                        st.metric("Confidence Score", confidence)

    elif page == "Reports":
        st.header("8. Reports")
        if st.session_state.metrics_table is None:
            st.warning("Train a model first to generate reports.")
        else:
            st.subheader("Metrics Overview")
            st.dataframe(st.session_state.metrics_table, use_container_width=True)
            csv_bytes = st.session_state.metrics_table.to_csv(index=False).encode("utf-8")
            st.download_button("Download Metrics CSV", csv_bytes, file_name="model_metrics.csv", mime="text/csv")

            pdf_bytes = build_pdf_report(
                st.session_state.df_raw,
                st.session_state.task_type,
                st.session_state.target_column,
                st.session_state.metrics_table,
                st.session_state.best_model_name,
                st.session_state.preprocessing_report or {},
            )
            st.download_button("Download PDF Report", pdf_bytes, file_name="training_report.pdf", mime="application/pdf")

    elif page == "Model Management":
        st.header("9. Model Management")
        if st.session_state.model_artifact is None:
            st.warning("Train a model first to save and download it.")
        else:
            st.write(f"Saved model: {st.session_state.best_model_name}")
            st.download_button(
                "Download Model (.pkl)",
                st.session_state.model_artifact,
                file_name="trained_model.pkl",
                mime="application/octet-stream",
            )

    st.sidebar.markdown("---")
    st.sidebar.write("Built with Streamlit, Plotly, Scikit-learn and FPDF.")


if __name__ == "__main__":
    main()
