import argparse, numpy as np, pandas as pd
rng = np.random.default_rng(42)

def synthesize(n=5000):
    # Toy SaaS-like features
    months = rng.integers(1, 36, n)
    plan = rng.choice(["basic","pro","enterprise"], n, p=[0.6, 0.35, 0.05])
    seats = (plan == "enterprise")*rng.integers(10,100,n) + (plan == "pro")*rng.integers(3,20,n) + (plan=="basic")*rng.integers(1,5,n)
    usage_7d = rng.gamma(2, 5, n) * (plan=="pro") + rng.gamma(3, 6, n) * (plan=="enterprise") + rng.gamma(1.8, 4, n)*(plan=="basic")
    tickets_30d = rng.poisson( (plan=="basic")*1.2 + (plan=="pro")*0.8 + (plan=="enterprise")*0.5, n )
    nps = rng.normal(30, 20, n) - (plan=="basic")*10 + (plan=="enterprise")*5
    discount = rng.choice([0,10,20], n, p=[0.7,0.2,0.1])
    price = (plan=="basic")*20 + (plan=="pro")*60 + (plan=="enterprise")*200
    mrr = price * np.clip(seats,1,None) * (1 - discount/100)

    # churn probability (nonlinear)
    logits = (
        -2.0
        - 0.03*months
        - 0.001*usage_7d
        + 0.15*tickets_30d
        - 0.01*nps
        - 0.0005*mrr
        + (plan=="basic")*0.5
    )
    p = 1/(1+np.exp(-logits))
    churn = rng.binomial(1, p)
    df = pd.DataFrame({
        "months_active":months, "plan":plan, "seats":seats, "usage_7d":usage_7d,
        "tickets_30d":tickets_30d, "nps":nps, "discount_pct":discount, "mrr":mrr, "churn":churn
    })
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw/churn.csv")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    df = synthesize(args.n)
    df.to_csv(args.out, index=False)
    print(f"saved {args.out}, shape={df.shape}")
