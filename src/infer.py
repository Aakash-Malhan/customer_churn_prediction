import argparse, joblib, pandas as pd
from .features import split_Xy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True, help="CSV with same columns (no churn needed)")
    ap.add_argument("--out", type=str, default="scored.csv")
    args = ap.parse_args()

    pipe = joblib.load("models/artifacts/model.joblib")
    df = pd.read_csv(args.inp)
    # If churn column exists, ignore in scoring
    cols = [c for c in df.columns if c != "churn"]
    prob = pipe.predict_proba(df[cols])[:,1]
    out = df.assign(churn_prob=prob).sort_values("churn_prob", ascending=False)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}, head:\n", out.head(5))
