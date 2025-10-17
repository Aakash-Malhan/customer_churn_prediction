import argparse, json, joblib, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from .features import build_preprocess, split_Xy

def recall_at_topk(y_true, y_prob, k=0.05):
    n_top = max(1, int(len(y_true)*k))
    idx = np.argsort(-y_prob)[:n_top]
    return (y_true[idx].sum()) / max(1, y_true.sum())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="data/raw/churn.csv")
    ap.add_argument("--k", type=float, default=0.05, help="top-k fraction for recall@k")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    X, y = split_Xy(df)
    pre = build_preprocess()

    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss"
    )
    model = CalibratedClassifierCV(xgb, cv=3, method="isotonic")

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("pre", pre), ("clf", model)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)

    prob = pipe.predict_proba(Xte)[:,1]
    pr_auc = average_precision_score(yte, prob)
    r_at_k = recall_at_topk(yte, prob, k=args.k)
    prec, rec, thr = precision_recall_curve(yte, prob)

    metrics = {"pr_auc": float(pr_auc), f"recall_at_top_{args.k:.0%}": float(r_at_k)}
    print(metrics)

    import os, json
    os.makedirs("models/artifacts", exist_ok=True)
    os.makedirs("models/metrics", exist_ok=True)
    joblib.dump(pipe, "models/artifacts/model.joblib")
    with open("models/metrics/metrics.json","w") as f: json.dump(metrics,f,indent=2)
