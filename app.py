import streamlit as st, pandas as pd, joblib
from pathlib import Path

st.title("Customer Churn – Save List")
st.write("Upload a CSV (same schema as training). We'll rank customers by churn probability and show drivers.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if st.button("Use demo data"):
    import subprocess, sys
    subprocess.run([sys.executable, "src/make_dataset.py", "--out", "data/raw/churn.csv"])
    uploaded = "data/raw/churn.csv"

if uploaded:
    if isinstance(uploaded, str):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(uploaded)

    if not Path("models/artifacts/model.joblib").exists():
        st.warning("No model found. Training a quick one now...")
        import subprocess, sys
        subprocess.run([sys.executable, "src/train.py", "--in", "data/raw/churn.csv"])

    pipe = joblib.load("models/artifacts/model.joblib")
    prob = pipe.predict_proba(df.drop(columns=[c for c in ["churn"] if c in df.columns]))[:,1]
    scored = df.copy()
    scored["churn_prob"] = prob
    st.dataframe(scored.sort_values("churn_prob", ascending=False).head(50))

    # SHAP summary for top rows (optional light compute)
    try:
        import shap
        explainer = shap.Explainer(pipe.named_steps["clf"], feature_names=None)
        # For tree models with pipeline preprocessing, fallback to permutation
        st.write("Feature importance (global):")
        import numpy as np
        import matplotlib.pyplot as plt
        shap_vals = None
        with st.spinner("Computing permutation importances..."):
            # Simple permutation importance via sklearn if shap fails
            from sklearn.inspection import permutation_importance
            X = df.drop(columns=[c for c in ["churn"] if c in df.columns]).iloc[:500]
            y = None
            result = permutation_importance(pipe, X, pipe.predict(X), n_repeats=3, random_state=42)
            imp = pd.DataFrame({"feature": X.columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)
            st.dataframe(imp.head(15))
    except Exception as e:
        st.info(f"Explainability fallback: {e}")
