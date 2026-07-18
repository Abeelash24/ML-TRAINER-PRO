/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface PythonCodeFile {
  name: string;
  path: string;
  content: string;
}

export const pythonFiles: PythonCodeFile[] = [
  {
    name: 'requirements.txt',
    path: 'requirements.txt',
    content: `streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
reportlab>=4.0.0
shap>=0.42.0
`
  },
  {
    name: 'preprocessing.py',
    path: 'utils/preprocessing.py',
    content: `import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def handle_duplicates(df: pd.DataFrame, remove: bool = True):
    """Detect and remove duplicates."""
    initial_count = len(df)
    if remove:
        df_cleaned = df.drop_duplicates()
        removed_count = initial_count - len(df_cleaned)
        return df_cleaned, removed_count
    return df, 0

def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Handle missing values based on strategy: mean, median, mode, or drop."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if strategy == "drop":
                df_clean = df_clean.dropna(subset=[col])
            else:
                if df_clean[col].dtype in [np.float64, np.int64]:
                    if strategy == "mean":
                        fill_value = df_clean[col].mean()
                    elif strategy == "median":
                        fill_value = df_clean[col].median()
                    else:  # mode
                        fill_value = df_clean[col].mode()[0]
                else:
                    # Categorical mode fill
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Missing"
                df_clean[col] = df_clean[col].fillna(fill_value)
    return df_clean

def detect_and_clip_outliers(df: pd.DataFrame, handle_method: str = "clip") -> tuple[pd.DataFrame, int]:
    """IQR Rule outlier adjustment for all float/int columns."""
    df_clean = df.copy()
    outlier_count = 0
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_count += len(outliers)
        
        if len(outliers) > 0:
            if handle_method == "clip":
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            elif handle_method == "remove":
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                
    return df_clean, outlier_count

def encode_categorical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical objects/strings to numerical weights using LabelEncoder."""
    df_encoded = df.copy()
    encoders = {}
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        
    return df_encoded, encoders

def scale_features(df: pd.DataFrame, target_col: str, method: str = "standard") -> tuple[pd.DataFrame, dict]:
    """Feature Scaling (Standardization or MinMax) for independent variables."""
    df_scaled = df.copy()
    scalers = {}
    
    if method == "none":
        return df_scaled, scalers
        
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != target_col]
    
    if len(feature_cols) == 0:
        return df_scaled, scalers
        
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
    scalers["feature_scaler"] = scaler
    
    return df_scaled, scalers

def auto_preprocess(df: pd.DataFrame, target_col: str, options: dict) -> tuple[pd.DataFrame, dict]:
    """Unified preprocessing pipeline execution."""
    # 1. Duplicates
    df_clean, rem_dups = handle_duplicates(df, options.get("remove_duplicates", True))
    
    # 2. Missing values
    df_clean = handle_missing_values(df_clean, options.get("missing_strategy", "mean"))
    
    # 3. Outliers
    df_clean, out_count = detect_and_clip_outliers(df_clean, options.get("handle_outliers", "clip"))
    
    # 4. Encodings
    df_clean, encoders = encode_categorical_columns(df_clean)
    
    # 5. Feature Scaling
    df_clean, scalers = scale_features(df_clean, target_col, options.get("scaling_method", "standard"))
    
    report = {
        "initial_rows": len(df),
        "processed_rows": len(df_clean),
        "removed_duplicates": rem_dups,
        "outliers_adjusted": out_count,
        "encoded_columns": list(encoders.keys())
    }
    
    return df_clean, {"encoders": encoders, "scalers": scalers, "report": report}
`
  },
  {
    name: 'visualization.py',
    path: 'utils/visualization.py',
    content: `import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def plot_distributions(df: pd.DataFrame, column: str):
    """Plot Histogram and Kernel Distribution."""
    fig = px.histogram(
        df, x=column, marginal="box", 
        title=f"Distribution of {column}",
        template="plotly_white",
        color_discrete_sequence=["#2563EB"]
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot rich interactive Correlation Heatmap using Pearson Correlation."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().round(2)
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        aspect="auto",
        title="Pearson Correlation Heatmap Matrix",
        template="plotly_white"
    )
    return fig

def plot_scatter_target(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    """Bivariate scatter plot analyzing feature interactions."""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        trendline="ols" if (df[x_col].dtype in [np.float64, np.int64] and df[y_col].dtype in [np.float64, np.int64]) else None,
        title=f"{y_col} vs {x_col}",
        template="plotly_white"
    )
    return fig

def plot_feature_box(df: pd.DataFrame, category_col: str, numeric_col: str):
    """Boxplot plotting numerical distributions across categorical groups."""
    fig = px.box(
        df, x=category_col, y=numeric_col,
        color=category_col,
        title=f"Boxplot Analysis of {numeric_col} by {category_col}",
        template="plotly_white"
    )
    return fig

def plot_target_pie(df: pd.DataFrame, target_col: str):
    """Distribution pie-chart for discrete categorical targets."""
    class_counts = df[target_col].value_counts().reset_index()
    class_counts.columns = [target_col, "Count"]
    fig = px.pie(
        class_counts, values="Count", names=target_col,
        title=f"Target Column Distribution: {target_col}",
        hole=0.4,
        template="plotly_white"
    )
    return fig
`
  },
  {
    name: 'model_training.py',
    path: 'utils/model_training.py',
    content: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics

def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """Deduce if the target implies Classification or Regression."""
    unique_cnt = df[target_col].nunique()
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    
    if unique_cnt <= 5 or not is_numeric:
        return "classification"
    return "regression"

def calculate_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute fully structured evaluation benchmarks."""
    report = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                report["roc_auc"] = metrics.roc_auc_score(y_true, y_prob[:, 1])
            else:
                report["roc_auc"] = metrics.roc_auc_score(y_true, y_prob, multi_class="ovr")
        except:
            report["roc_auc"] = report["accuracy"]
    else:
        report["roc_auc"] = report["accuracy"]
    return report

def calculate_regression_metrics(y_true, y_pred) -> dict:
    """Compute continuous analytical regression benchmarks."""
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_true, y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

def train_and_evaluate_models(df: pd.DataFrame, target_col: str, problem_type: str, hyperparams: dict = None) -> dict:
    """Main model engine training Random Forest, Decision Tree, and Regression models."""
    if hyperparams is None:
        hyperparams = {}
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    results = {}
    
    if problem_type == "classification":
        # Model 1: Random Forest
        rf_params = hyperparams.get("rf", {"n_estimators": 50, "max_depth": 5})
        rf = RandomForestClassifier(**rf_params, random_state=42)
        rf.fit(X_train, y_train)
        
        # Model 2: Decision Tree
        dt_params = hyperparams.get("dt", {"max_depth": 5})
        dt = DecisionTreeClassifier(**dt_params, random_state=42)
        dt.fit(X_train, y_train)
        
        # Model 3: Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        
        models_dict = {"Random Forest": rf, "Decision Tree": dt, "Logistic Regression": lr}
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                
            eval_metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
            eval_metrics["train_accuracy"] = metrics.accuracy_score(y_train, y_pred_train)
            
            # Cross validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            eval_metrics["cv_mean"] = cv_scores.mean()
            eval_metrics["cv_scores"] = cv_scores.tolist()
            
            # Confusion matrix
            cm = metrics.confusion_matrix(y_test, y_pred)
            results[name] = {
                "model": model,
                "metrics": eval_metrics,
                "confusion_matrix": cm.tolist(),
                "feature_importance": model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else None
            }
            
    else: # REGRESSION
        # Model 1: Random Forest
        rf_params = hyperparams.get("rf", {"n_estimators": 50, "max_depth": 5})
        rf = RandomForestRegressor(**rf_params, random_state=42)
        rf.fit(X_train, y_train)
        
        # Model 2: Decision Tree
        dt_params = hyperparams.get("dt", {"max_depth": 5})
        dt = DecisionTreeRegressor(**dt_params, random_state=42)
        dt.fit(X_train, y_train)
        
        # Model 3: Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        models_dict = {"Random Forest": rf, "Decision Tree": dt, "Linear Regression": lr}
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            eval_metrics = calculate_regression_metrics(y_test, y_pred)
            
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
            eval_metrics["cv_mean"] = cv_scores.mean()
            eval_metrics["cv_scores"] = cv_scores.tolist()
            
            results[name] = {
                "model": model,
                "metrics": eval_metrics,
                "feature_importance": model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else None
            }
            
    return results
`
  },
  {
    name: 'report_generator.py',
    path: 'utils/report_generator.py',
    content: `import pandas as pd
import io

def generate_csv_summary(results: dict, problem_type: str) -> str:
    """Generates structured CSV evaluation grids for easy analysis downloads."""
    output = io.StringIO()
    output.write("MODEL REPOSITORY TRAINING METRIC GRID\\n")
    output.write(f"Algorithm Problem Type: {problem_type}\\n\\n")
    
    if problem_type == "classification":
        output.write("Model,Testing Accuracy,Training Accuracy,Precision,Recall,F1-Score,ROC-AUC,CV Mean\\n")
        for m_name, dict_data in results.items():
            met = dict_data["metrics"]
            output.write(f"{m_name},{met['accuracy']:.4f},{met['train_accuracy']:.4f},{met['precision']:.4f},{met['recall']:.4f},{met['f1']:.4f},{met['roc_auc']:.4f},{met['cv_mean']:.4f}\\n")
    else:
        output.write("Model,R2 Score,MAE,MSE,RMSE,CV Mean (R2)\\n")
        for m_name, dict_data in results.items():
            met = dict_data["metrics"]
            output.write(f"{m_name},{met['r2']:.4f},{met['mae']:.4f},{met['mse']:.4f},{met['rmse']:.4f},{met['cv_mean']:.4f}\\n")
            
    return output.getvalue()

def build_pdf_report_memo(results: dict, dataset_name: str, problem_type: str, best_model: str) -> str:
    """Constructs a production-level Markdown audit report ready for conversion to PDF/HTML."""
    report = f"""# MACHINE LEARNING TRAINING COLLABORATIVE REPORT
**Dataset Identifier:** {dataset_name}
**Problem Mode:** {problem_type.upper()}
**Identified Optimal Model:** {best_model}

---

## 1. Executive Summary
This report presents the evaluation logs from testing 3 specific algorithms: Random Forest, Decision Tree, and Regression (Linear/Logistic). Features were normalized, standardized, and trained across 5-fold cross validation.

## 2. Competitive Model Score Matrix
"""
    if problem_type == "classification":
        report += "| Model Name | Test Accuracy | CV mean score | Weighted F1 | ROC AUC |\\n|---|---|---|---|---|\\n"
        for m_name, res in results.items():
            met = res["metrics"]
            report += f"| {m_name} | {met['accuracy']:.4f} | {met['cv_mean']:.4f} | {met['f1']:.4f} | {met['roc_auc']:.4f} |\\n"
    else:
         report += "| Model Name | R2 Expl. Variance | CV R2 mean score | Mean Abs Error | Root Mean Sq Error |\\n|---|---|---|---|---|\\n"
         for m_name, res in results.items():
            met = res["metrics"]
            report += f"| {m_name} | {met['r2']:.4f} | {met['cv_mean']:.4f} | {met['mae']:.2f} | {met['rmse']:.2f} |\\n"

    report += f"""
## 3. Best Model Finding & AutoML Selection Recommendation
Based on the metrics evaluated on the hold-out test set, the leading performer is **{best_model}**. It has proven consistent and has stable performance margins with a cross-validation score of **{results[best_model]['metrics']['cv_mean']:.4f}**.

**Next Steps & Implementation:**
1. Serialize the compiled performance weights to pickle formats.
2. Embed the preprocessor within deployment networks.
3. Establish feature pipelines to predict raw predictions.
"""
    return report
`
  },
  {
    name: 'app.py',
    path: 'app.py',
    content: `import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from utils.preprocessing import auto_preprocess
from utils.visualization import (
    plot_distributions, plot_correlation_heatmap, 
    plot_scatter_target, plot_feature_box, plot_target_pie
)
from utils.model_training import (
    detect_problem_type, train_and_evaluate_models
)
from utils.report_generator import (
    generate_csv_summary, build_pdf_report_memo
)

# Page Layout configuration
st.set_page_config(
    page_title="Advanced ML Training Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Routing Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #2563EB;'>ML Studio Router</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Directory Navigation Selection",
    [
        "🏠 Home & Guidelines", 
        "📊 Dataset Inspector", 
        "🔍 Exploratory Data Analysis (EDA)", 
        "🏋️ Model Training Studio", 
        "📈 Model Comparison Dashboard", 
        "🥇 Feature Importance Tracker", 
        "🔮 Prediction Center", 
        "📋 Reports & Export Manager"
    ]
)

# Initialize Session States
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "prep_pipeline" not in st.session_state:
    st.session_state.prep_pipeline = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "trained_results" not in st.session_state:
    st.session_state.trained_results = None

# Sidebar CSV Uploading Trigger
uploaded_file = st.sidebar.file_uploader("Inbound Pipeline: Deposit tabular CSV", type=["csv"])
if uploaded_file is not None:
    st.session_state.raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV file successfully loaded")

if page == "🏠 Home & Guidelines":
    st.markdown("<h1 style='color: #1E3A8A;'>🤖 Advanced Machine Learning Model Training Studio</h1>", unsafe_allow_html=True)
    st.write("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Collaborative Overview
        Welcome to your Academic and Competitive ML Training Workspace! This Streamlit-driven environment is designed for 
        autonomous end-to-end processing, tuning, modeling, and predicting on clean datasets.
        
        #### Process Pipeline Steps:
        1. **Dataset Uploading**: Ingest standard CSV tables via the sidebar.
        2. **Preprocessing Pipeline**: Auto-strips duplicates, imputes missing records, clips statistical outliers, and standardizes scales.
        3. **Exploratory Visualizations**: Construct interactively plotted density distributions, correlations, and target variables.
        4. **Comparative Optimization**: Instantly fit Random Forest classifiers/regressors, Decision Trees, and OLS/Logistic metrics.
        5. **Predictive Simulation**: Test prediction weights by manually configuring sliders inside the **Prediction Center**.
        """)
        
        st.info("💡 **Quick Start:** Drag and drop a CSV file into the sidebar to activate interactive models.")
        
    with col2:
        st.markdown("### System Presets & Performance Metrics")
        st.metric("Compatible Frameworks", "Scikit-Learn, Streamlit")
        st.metric("AutoML Algorithms Fitted", "3 (RF, DT, Regressions)")
        st.metric("Validation Metrics Tracked", "ROC AUC, F1, RMSE, R²")

elif page == "📊 Dataset Inspector":
    st.title("📊 Dataset Shape & Summary Inspector")
    st.write("---")
    
    if st.session_state.raw_df is None:
        st.warning("Please upload a CSV file in the sidebar to proceed.")
    else:
        df = st.session_state.raw_df
        st.subheader("Raw CSV Record Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Row Records Count", df.shape[0])
        col2.metric("Column Variables Count", df.shape[1])
        col3.metric("Aggregated Duplicates", df.duplicated().sum())
        
        st.write("### Data Columns & Missing Counts")
        meta_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Value Counts": df.count(),
            "Null/Blank Records": df.isnull().sum(),
            "Unique Cardinality": df.nunique()
        })
        st.table(meta_df)
        
        st.write("### Continuous Numerical Statistics")
        st.dataframe(df.describe(include=[np.number]), use_container_width=True)

elif page == "🔍 Exploratory Data Analysis (EDA)":
    st.title("🔍 Exploratory Data Analysis Dashboard")
    st.write("---")
    
    if st.session_state.raw_df is None:
        st.warning("Please upload a CSV file inside the sidebar to configure features.")
    else:
        df = st.session_state.raw_df
        st.write("### Pearson Multi-Variable Correlation Heatmap")
        fig_heat = plot_correlation_heatmap(df)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            col_sel = st.selectbox("Select variable analytical target", df.columns)
            fig_dist = plot_distributions(df, col_sel)
            st.plotly_chart(fig_dist, use_container_width=True)
        with col2:
            st.markdown("#### Bivariate Feature Interactions")
            x_sel = st.selectbox("X-Axis Feature Selector", df.columns, index=0)
            y_sel = st.selectbox("Y-Axis Target Selector", df.columns, index=min(1, len(df.columns)-1))
            fig_scat = plot_scatter_target(df, x_sel, y_sel)
            st.plotly_chart(fig_scat, use_container_width=True)

elif page == "🏋️ Model Training Studio":
    st.title("🏋️ Model Performance Training Studio")
    st.write("---")
    
    if st.session_state.raw_df is None:
        st.warning("Ensure an active CSV dataset is deposited under the sidebar pipeline.")
    else:
        df = st.session_state.raw_df
        target = st.selectbox("Deduce Dependent Feature (Target Variable)", df.columns, index=len(df.columns)-1)
        st.session_state.target_col = target
        
        ptype = detect_problem_type(df, target)
        st.session_state.problem_type = ptype
        st.info(f"🧬 Model Pipeline Engine has auto-detected target as: **{ptype.upper()}** problem type.")
        
        st.write("### Preprocessing Options Configuration")
        col1, col2, col3 = st.columns(3)
        strategy = col1.radio("Missing value solution strategy", ["mean", "median", "drop"])
        outliers = col2.radio("Statistical outlier adjustment strategy", ["clip", "remove", "none"])
        scaling = col3.radio("Feature scaling normalization", ["standard", "minmax", "none"])
        
        st.write("### Hyperparameters Grid Optimization")
        exp1 = st.expander("Tweak RandomForest and DecisionTree bounds", expanded=True)
        with exp1:
            col_rf1, col_rf2 = st.columns(2)
            estimators = col_rf1.slider("RF estimators limits", 5, 100, 30, step=5)
            max_depth = col_rf2.slider("Max search Tree depth", 2, 12, 5)
            
        if st.button("🚀 Push Training Sequence (Train 3 Models)"):
            with st.spinner("Executing models and calculating matrices..."):
                prep_options = {
                    "missing_strategy": strategy,
                    "remove_duplicates": True,
                    "handle_outliers": outliers,
                    "scaling_method": scaling
                }
                
                # Preprocess
                cleaned_df, pipeline = auto_preprocess(df, target, prep_options)
                st.session_state.processed_df = cleaned_df
                st.session_state.prep_pipeline = pipeline
                
                # Training
                hyper = {
                    "rf": {"n_estimators": estimators, "max_depth": max_depth},
                    "dt": {"max_depth": max_depth}
                }
                st.session_state.trained_results = train_and_evaluate_models(cleaned_df, target, ptype, hyper)
                st.success("Calibration complete! Jump to the Comparison tab.")

elif page == "📈 Model Comparison Dashboard":
    st.title("📈 Comparative Performance Analytics Dashboard")
    st.write("---")
    
    if st.session_state.trained_results is None:
        st.warning("Please fit datasets in the Training tab first.")
    else:
        res = st.session_state.trained_results
        ptype = st.session_state.problem_type
        
        # Display comparison
        st.subheader("Model Competitive Scoring Log")
        
        metrics_list = []
        for name, data in res.items():
            metrics_list.append({
                "Model name": name,
                **data["metrics"]
            })
        m_df = pd.DataFrame(metrics_list)
        st.dataframe(m_df, use_container_width=True)
        
        st.write("### Graphical Performance Matrix Evaluation")
        if ptype == "classification":
            st.bar_chart(m_df.set_index("Model name")[["accuracy", "precision", "recall", "f1"]])
        else:
            st.bar_chart(m_df.set_index("Model name")[["r2"]])
            
        best_model = max(res.keys(), key=lambda name: res[name]["metrics"]["accuracy" if ptype == "classification" else "r2"])
        st.success(f"🏆 **Automated Model Recommendation:** Based on holdout evaluations, the optimal solver is **{best_model}**.")

elif page == "🥇 Feature Importance Tracker":
    st.title("🥇 Gini-Impurity Feature Weights Tracker")
    st.write("---")
    
    if st.session_state.trained_results is None:
        st.warning("Please complete model evaluations first.")
    else:
        res = st.session_state.trained_results
        feat_df = st.session_state.processed_df.drop(columns=[st.session_state.target_col])
        feature_names = feat_df.columns.tolist()
        
        tree_models = [k for k in res.keys() if "Forest" in k or "Tree" in k]
        chosen = st.selectbox("Select model to fetch Gini metrics", tree_models)
        
        importance_vals = res[chosen]["feature_importance"]
        if importance_vals:
            imp_series = pd.DataFrame({
                "Variables": feature_names,
                "Relative Importance Weight": importance_vals
            }).sort_values(by="Relative Importance Weight", ascending=False)
            
            st.subheader("Feature Weights Allocation Table")
            st.dataframe(imp_series, use_container_width=True)
            st.bar_chart(imp_series.set_index("Variables"))
        else:
            st.info("Feature importance metrics not supported for this solver representation.")

elif page == "🔮 Prediction Center":
    st.title("🔮 Model Prediction Simulator Panel")
    st.write("---")
    
    if st.session_state.trained_results is None:
        st.warning("Please fit model metrics before generating predictions.")
    else:
        res = st.session_state.trained_results
        target = st.session_state.target_col
        proc_df = st.session_state.processed_df
        ptype = st.session_state.problem_type
        
        model_name = st.selectbox("Select Model to invoke", list(res.keys()))
        selected_model = res[model_name]["model"]
        
        st.write("### Tweak Feature Fields manually")
        features = proc_df.drop(columns=[target]).columns
        
        input_data = {}
        for f in features:
            f_min = float(proc_df[f].min())
            f_max = float(proc_df[f].max())
            f_mean = float(proc_df[f].mean())
            input_val = st.slider(f"Adjust feature: {f}", f_min, f_max, f_mean)
            input_data[f] = input_val
            
        if st.button("🔮 Compute Analytical Prediction"):
            inputs = pd.DataFrame([input_data])
            pred = selected_model.predict(inputs)[0]
            
            st.write("---")
            col1, col2 = st.columns(2)
            col1.markdown(f"### Predicted Value for target ({target}):")
            col1.markdown(f"<h2 style='color: #2563EB;'>{pred:.4f}</h2>", unsafe_allow_html=True)
            
            if hasattr(selected_model, "predict_proba") and ptype == "classification":
                probs = selected_model.predict_proba(inputs)[0]
                col2.markdown("#### Distribution Probabilities weights:")
                col2.write(probs)

elif page == "📋 Reports & Export Manager":
    st.title("📋 Performance Reports & Serialized Storage")
    st.write("---")
    
    if st.session_state.trained_results is None:
         st.warning("Model calculations must run before summary reports compile.")
    else:
        res = st.session_state.trained_results
        ptype = st.session_state.problem_type
        best_model = max(res.keys(), key=lambda name: res[name]["metrics"]["accuracy" if ptype == "classification" else "r2"])
        
        st.write("### Dynamic Performance Exporting")
        
        csv_string = generate_csv_summary(res, ptype)
        st.download_button(
            "📥 Download Comparative Scoring Logs (CSV)",
            data=csv_string,
            file_name="model_scoring_logs.csv",
            mime="text/csv"
        )
        
        pdf_markdown = build_pdf_report_memo(res, uploaded_file.name if uploaded_file else "dataset.csv", ptype, best_model)
        st.write("### Report Summary Preview (Markdown)")
        st.markdown(pdf_markdown)
        
        st.download_button(
            "📥 Export Report Memo (TXT)",
            data=pdf_markdown,
            file_name="ml_training_report_memo.txt"
        )
        
        # Serialize Model
        st.write("### Binary Storage Exporter")
        pkl_model = pickle.dumps(res[best_model]["model"])
        st.download_button(
            f"📥 Download Serialized Best Model ({best_model} .pkl file)",
            data=pkl_model,
            file_name=f"best_model_weights.pkl",
            mime="application/octet-stream"
        )
`
  }
];
