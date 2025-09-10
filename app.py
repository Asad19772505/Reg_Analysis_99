import os
import json
import time
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import matplotlib.pyplot as plt
import io  # for Excel export

# Optional libs
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Time series
from statsmodels.tsa.statespace.sarimax import SARIMAX

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def _jsonable(obj):
    """Recursively convert objects (np types, pandas, timestamps) to JSON-safe python types."""
    import numpy as _np
    import pandas as _pd
    import datetime as _dt

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _pd.Timestamp):
        return obj.to_pydatetime().isoformat()
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, _pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, _pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return None

def _df_download_buttons(df: pd.DataFrame, base_name: str, label_prefix: str = "Download"):
    """Render Streamlit buttons to download a DataFrame as CSV or Excel."""
    if df is None or df.empty:
        return
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label=f"{label_prefix} CSV",
        data=csv_bytes,
        file_name=f"{base_name}.csv",
        mime="text/csv",
        use_container_width=True
    )
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=True, sheet_name="data")
    xbuf.seek(0)
    st.download_button(
        label=f"{label_prefix} Excel",
        data=xbuf.getvalue(),
        file_name=f"{base_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.set_page_config(page_title="Finance ML Workbench", layout="wide")
st.title("🧭 Finance ML Workbench — Select a Problem Type")

PROBLEM_TYPES = [
    "Regression",
    "Classification",
    "Clustering",
    "Time Series Forecasting",
    "Anomaly Detection",
    "Recommendation System (stub)",
    "Optimization (stub)",
]

pt = st.sidebar.selectbox("Problem Type", PROBLEM_TYPES)
st.sidebar.info("Upload a CSV and configure options for the selected problem type.")

uploaded = st.file_uploader("Upload CSV data", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("### Data Preview", df.head())

def split_features(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = list(set(X.columns) - set(num_cols))
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return X, y, pre

def save_artifacts(name, model=None, metrics=None, extra=None):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(ARTIFACT_DIR, f"{name}_{stamp}")
    os.makedirs(base, exist_ok=True)
    if model is not None:
        joblib.dump(model, os.path.join(base, "model.pkl"))
    if metrics is not None:
        with open(os.path.join(base, "metrics.json"), "w") as f:
            json.dump(_jsonable(metrics), f, indent=2)
    if extra is not None:
        with open(os.path.join(base, "extra.json"), "w") as f:
            json.dump(_jsonable(extra), f, indent=2)
    st.success(f"Artifacts saved to `{base}`")
    return base

# ---------------- Regression ----------------
def run_regression(df):
    target = st.selectbox("Select target (numeric)", df.select_dtypes(include=[np.number]).columns)
    algo_choices = ["LinearRegression", "RandomForestRegressor"]
    if XGB_AVAILABLE:
        algo_choices.append("XGBoostRegressor")
    else:
        st.caption("Tip: `pip install xgboost` to enable XGBoost.")
    algo = st.selectbox("Algorithm", algo_choices)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    if st.button("Train Regression"):
        X, y, pre = split_features(df, target)
        if algo == "LinearRegression":
            model = Pipeline([("pre", pre), ("clf", LinearRegression())])
        elif algo == "RandomForestRegressor":
            model = Pipeline([("pre", pre), ("clf", RandomForestRegressor(n_estimators=300, random_state=42))])
        else:
            model = Pipeline([("pre", pre), ("clf", XGBRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                n_jobs=-1, objective="reg:squarederror"))])

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        mse = mean_squared_error(yte, pred)       # cross-version safe
        rmse = float(math.sqrt(mse))
        r2 = r2_score(yte, pred)
        st.write({"MAE": mae, "RMSE": rmse, "R2": r2})
        save_artifacts("regression", model, {"MAE": mae, "RMSE": rmse, "R2": r2})

# ---------------- Classification ----------------
def run_classification(df):
    target = st.selectbox("Select target (categorical/binary)", df.columns)
    algo_choices = ["LogisticRegression", "RandomForestClassifier", "SVM (linear)"]
    if XGB_AVAILABLE:
        algo_choices.append("XGBoostClassifier")
    else:
        st.caption("Tip: `pip install xgboost` to enable XGBoost.")
    algo = st.selectbox("Algorithm", algo_choices)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    if st.button("Train Classification"):
        y = df[target].astype("category")
        df2 = df.copy()
        df2[target] = y.cat.codes
        X, y, pre = split_features(df2, target)

        if algo == "LogisticRegression":
            clf = LogisticRegression(max_iter=500)
        elif algo == "RandomForestClassifier":
            clf = RandomForestClassifier(n_estimators=300, random_state=42)
        elif algo == "SVM (linear)":
            clf = SVC(kernel="linear", probability=True)
        else:
            clf = XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
                objective="multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic",
                eval_metric="logloss"
            )

        model = Pipeline([("pre", pre), ("clf", clf)])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        prec = precision_score(yte, pred, average="weighted", zero_division=0)
        rec = recall_score(yte, pred, average="weighted", zero_division=0)
        f1 = f1_score(yte, pred, average="weighted", zero_division=0)
        st.write({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        st.write("Confusion Matrix")
        st.dataframe(pd.DataFrame(confusion_matrix(yte, pred)))
        save_artifacts("classification", model, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

# ---------------- Clustering ----------------
def run_clustering(df):
    X = df.select_dtypes(include=[np.number]).copy()
    st.caption("Clustering uses numeric features only.")
    algo = st.selectbox("Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])
    if algo == "KMeans":
        k = st.slider("k (clusters)", 2, 12, 4)
    elif algo == "DBSCAN":
        eps = st.number_input("eps", value=0.5, min_value=0.05, step=0.05)
        min_samples = st.number_input("min_samples", value=5, min_value=2, step=1)
    else:
        k = st.slider("n_clusters", 2, 12, 4)

    if st.button("Run Clustering"):
        Xs = StandardScaler().fit_transform(X)
        if algo == "KMeans":
            model = KMeans(n_clusters=k, random_state=42, n_init=10)  # cross-version safe
            labels = model.fit_predict(Xs)
        elif algo == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=int(min_samples))
            labels = model.fit_predict(Xs)
        else:
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(Xs)

        df_out = df.copy()
        df_out["cluster"] = labels
        st.write("Clustered Sample", df_out.head())

        unique_labels = set(labels)
        if len(unique_labels) > 1 and not (len(unique_labels) == 2 and (-1 in unique_labels)):
            try:
                st.write({"silhouette_score": silhouette_score(Xs, labels)})
            except Exception as e:
                st.info(f"Silhouette not available: {e}")
        save_artifacts("clustering", model, {"note": "completed"})

# ---------------- Time Series (duplicate-safe) ----------------
def _seasonal_periods(freq: str) -> int:
    if not isinstance(freq, str):
        return 12
    f = freq.upper().strip()
    return {"D": 7, "W": 52, "M": 12, "Q": 4, "A": 1, "Y": 1}.get(f, 12)

def _resample_series(df, date_col, target, freq, agg):
    """Resample to desired freq with aggregation to remove duplicates."""
    tmp = df[[date_col, target]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    s = tmp.groupby(pd.Grouper(key=date_col, freq=freq))[target].agg(agg)
    if len(s.index) > 0:
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq=freq)
        s = s.reindex(full_idx)
    return s

def run_timeseries(df):
    date_col = st.selectbox("Date column", df.columns)
    target = st.selectbox("Target (numeric)", df.select_dtypes(include=[np.number]).columns)
    freq = st.text_input("Pandas frequency (e.g., D, W, M)", "M")
    horizon = st.number_input("Forecast horizon (periods)", 3, 60, 12)
    agg = st.selectbox("Aggregation for duplicates/resampling", ["mean", "sum", "last"], index=0)
    fill_strategy = st.selectbox("Gap fill", ["interpolate", "ffill", "zero"], index=0)

    model_choices = ["SARIMAX"]
    if PROPHET_AVAILABLE: model_choices.append("Prophet (additive)")
    else: st.caption("Tip: `pip install prophet` to enable Prophet.")
    model_type = st.selectbox("Time-series model", model_choices)

    if st.button("Forecast"):
        s = _resample_series(df, date_col, target, freq, agg)

        if fill_strategy == "interpolate":
            s = s.interpolate(limit_direction="both")
        elif fill_strategy == "ffill":
            s = s.fillna(method="ffill").fillna(method="bfill")
        else:
            s = s.fillna(0.0)

        s = s.dropna()
        if len(s) < 8:
            st.error("Not enough points after resampling/filling. Provide more data or change frequency.")
            return

        if model_type.startswith("SARIMAX"):
            sp = _seasonal_periods(freq)
            res = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1, sp)).fit(disp=False)
            pred = res.get_forecast(steps=int(horizon)).predicted_mean
            out = pd.DataFrame({"y": s, "forecast": pred})
            st.line_chart(out)
            _df_download_buttons(out, base_name="sarimax_forecast", label_prefix="Download SARIMAX")
            st.write({"aic": float(res.aic), "bic": float(res.bic)})
            save_artifacts("timeseries_sarimax", None, {"aic": float(res.aic), "bic": float(res.bic)},
                           extra={"last_forecast": pred.tail(1).to_dict()})
        else:
            prophet_df = s.reset_index()
            prophet_df.columns = ["ds", "y"]
            m = Prophet(seasonality_mode="additive")
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=int(horizon), freq=freq)
            fcst = m.predict(future)
            merged = pd.merge(prophet_df, fcst[["ds", "yhat"]], on="ds", how="outer").set_index("ds").sort_index()
            merged.rename(columns={"y": "actual", "yhat": "forecast"}, inplace=True)
            st.line_chart(merged)
            _df_download_buttons(merged, base_name="prophet_forecast", label_prefix="Download Prophet")
            insample = merged.dropna()
            mae = float(np.mean(np.abs(insample["actual"] - insample["forecast"]))) if not insample.empty else None
            st.write({"MAE (in-sample)": mae})
            save_artifacts("timeseries_prophet", None, {"mae_insample": mae},
                           extra={"last_forecast": merged["forecast"].tail(1).to_dict()})

# ---------------- Anomaly Detection ----------------
def run_anomaly(df):
    X = df.select_dtypes(include=[np.number]).copy()
    algo = st.selectbox("Algorithm", ["IsolationForest", "OneClassSVM"])
    cont = st.slider("Contamination (expected outlier %)", 0.01, 0.2, 0.05, 0.01)
    if st.button("Detect Anomalies"):
        Xs = StandardScaler().fit_transform(X)
        model = IsolationForest(contamination=cont, random_state=42) if algo == "IsolationForest" else OneClassSVM(nu=cont, kernel="rbf")
        labels = model.fit_predict(Xs)  # -1 = outlier
        df_out = df.copy()
        df_out["is_outlier"] = (labels == -1).astype(int)
        st.write("Flagged data", df_out.head())
        st.metric("Outliers found", int(df_out["is_outlier"].sum()))
        save_artifacts("anomaly", model, {"outliers": int(df_out["is_outlier"].sum())})

# ---------------- Recommender (stub) ----------------
def run_recommender_stub(df):
    st.info("Expected columns: user_id, item_id, rating (0-5).")
    user_col = st.selectbox("User column", df.columns)
    item_col = st.selectbox("Item column", df.columns, index=min(1, len(df.columns)-1))
    rating_col = st.selectbox("Rating column", df.columns, index=min(2, len(df.columns)-1))
    target_user = st.text_input("User to recommend for", "")
    topn = st.slider("Top-N", 3, 20, 5)
    if st.button("Build & Recommend"):
        pivot = df.pivot_table(index=user_col, columns=item_col, values=rating_col, aggfunc="mean").fillna(0.0)
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            st.error("Need at least 2 users and 2 items.")
            return
        n_components = max(2, min(50, min(pivot.shape)-1))
        U = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(pivot)
        sim_df = pd.DataFrame(cosine_similarity(U), index=pivot.index, columns=pivot.index)
        if target_user not in sim_df.index:
            st.error("Target user not found.")
            return
        similar_users = sim_df[target_user].drop(target_user).sort_values(ascending=False).head(5).index
        recs = pivot.loc[similar_users].replace(0, np.nan).mean().dropna().sort_values(ascending=False).head(topn)
        st.write("Recommendations", recs)

# ---------------- Optimization (stub) ----------------
def run_optimization_stub(df):
    st.info("Expected columns: project, cost, roi; plus a Budget.")
    budget = st.number_input("Budget constraint", min_value=0.0, value=1000.0, step=100.0)
    try:
        from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD
    except Exception:
        st.error("Install pulp to enable optimization.")
        return
    proj_col = st.selectbox("Project column", df.columns)
    cost_col = st.selectbox("Cost column", df.columns)
    roi_col = st.selectbox("ROI column", df.columns)
    if st.button("Optimize Portfolio"):
        items = df[[proj_col, cost_col, roi_col]].dropna().copy()
        items.columns = ["proj", "cost", "roi"]
        m = LpProblem("BudgetAlloc", LpMaximize)
        x = {p: LpVariable(f"x_{i}", cat=LpBinary) for i, p in enumerate(items["proj"])}
        m += lpSum(x[p]*float(r) for p, r in zip(items["proj"], items["roi"]))
        m += lpSum(x[p]*float(c) for p, c in zip(items["proj"], items["cost"])) <= float(budget)
        m.solve(PULP_CBC_CMD(msg=False))
        chosen = [p for p in items["proj"] if x[p].value() == 1]
        st.write("Selected projects:", chosen)
        st.metric("Total ROI", float(sum(items.set_index("proj").loc[chosen]["roi"])) if chosen else 0.0)

# ---------------- Router ----------------
if uploaded is None:
    st.warning("Upload a CSV to begin.")
else:
    if pt == "Regression":
        run_regression(df)
    elif pt == "Classification":
        run_classification(df)
    elif pt == "Clustering":
        run_clustering(df)
    elif pt == "Time Series Forecasting":
        run_timeseries(df)
    elif pt == "Anomaly Detection":
        run_anomaly(df)
    elif pt == "Recommendation System (stub)":
        run_recommender_stub(df)
    elif pt == "Optimization (stub)":
        run_optimization_stub(df)
