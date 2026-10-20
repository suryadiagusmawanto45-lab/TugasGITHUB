"""Studi Kasus 1 - Make Sense of Data with Analytics and AI
Prediksi churn pelanggan: Descriptive -> Diagnostic -> Predictive analytics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "tenure_months": rng.integers(1, 60, n),
        "monthly_spend": rng.gamma(4, 25000, n).round(-2),
        "complaints": rng.poisson(1.0, n),
        "app_sessions": rng.poisson(12, n),
        "city": rng.choice(["Bengkulu", "Padang", "Jambi", "Palembang"], n, p=[.4, .2, .2, .2]),
    })
    logit = (-0.5 - 0.04 * df.tenure_months + 0.5 * df.complaints
             - 0.05 * df.app_sessions + 0.000003 * df.monthly_spend)
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def main():
    df = generate_data()
    df.to_csv(OUT / "customers.csv", index=False)

    # 1) Descriptive: apa yang terjadi?
    print("Churn rate keseluruhan: %.1f%%" % (df.churn.mean() * 100))
    by_city = df.groupby("city").churn.mean().mul(100).round(1)
    print("\nChurn rate per kota (%):\n", by_city)

    # 2) Diagnostic: mengapa terjadi?
    print("\nKorelasi fitur numerik dengan churn:")
    print(df.drop(columns=["customer_id", "city"]).corr()["churn"].drop("churn").round(3))

    # 3) Predictive: apa yang akan terjadi?
    X = pd.get_dummies(df.drop(columns=["customer_id", "churn"]), columns=["city"])
    y = df.churn
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    print("\nROC-AUC: %.3f" % roc_auc_score(yte, proba))
    # class_weight="balanced" menangani data tidak seimbang (churn hanya ~20%)
    print(classification_report(yte, model.predict(Xte)))

    # 4) Prescriptive (sederhana): daftar pelanggan risiko tinggi untuk retensi
    risk = Xte.copy()
    risk["churn_prob"] = proba
    risk.sort_values("churn_prob", ascending=False).head(20).to_csv(OUT / "top20_risiko_churn.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    by_city.plot.bar(ax=ax[0], color="#4c78a8", title="Churn rate per kota (%)")
    pd.Series(model.feature_importances_, X.columns).sort_values().plot.barh(
        ax=ax[1], color="#f58518", title="Feature importance")
    plt.tight_layout()
    plt.savefig(OUT / "analytics.png", dpi=150)
    print("Output tersimpan di", OUT)


if __name__ == "__main__":
    main()
