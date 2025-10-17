import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

NUM = ["months_active","seats","usage_7d","tickets_30d","nps","discount_pct","mrr"]
CAT = ["plan"]

def build_preprocess():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )

def split_Xy(df: pd.DataFrame):
    X = df[NUM + CAT].copy()
    y = df["churn"].astype(int).values
    return X, y
