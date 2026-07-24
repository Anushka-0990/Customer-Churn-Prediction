import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

try:
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    pass

from preprocess import get_processed_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def build_pipeline(model, numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, name, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.colorbar()
    plt.xticks([0, 1], ["Retained", "Churned"])
    plt.yticks([0, 1], ["Retained", "Churned"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150)
    plt.close()


def plot_feature_importance(pipeline, numeric_features, categorical_features):
    model = pipeline.named_steps["model"]
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_features = numeric_features + cat_names

    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": all_features, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(12)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2E86AB")
    plt.title("Top 12 Churn Drivers (Random Forest Feature Importance)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_feature_importance.png", dpi=150)
    plt.close()

    return imp_df


def generate_business_recommendations(imp_df: pd.DataFrame, df: pd.DataFrame) -> str:
    """Translate model + EDA findings into plain-English business recommendations."""
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    tenure_churn = df.groupby("tenure_group", observed=True)["Churn"].mean().sort_values(ascending=False)
    addon_churn = df.groupby("num_addon_services")["Churn"].mean()

    top_drivers = imp_df["feature"].head(5).tolist()

    lines = []
    lines.append("BUSINESS RECOMMENDATIONS — Customer Churn Reduction\n")
    lines.append("=" * 55 + "\n")

    lines.append(f"1. CONTRACT TYPE is a top churn driver.")
    lines.append(f"   Month-to-month customers churn at {contract_churn.iloc[0]*100:.1f}%, "
                  f"vs {contract_churn.iloc[-1]*100:.1f}% for the most stable contract type.")
    lines.append("   -> Recommendation: Offer a discounted incentive for month-to-month "
                  "customers to migrate to 1-year contracts (e.g., 5-10% loyalty discount).\n")

    lines.append(f"2. TENURE strongly predicts churn risk.")
    lines.append(f"   Customers in their first tenure bracket churn at {tenure_churn.iloc[0]*100:.1f}%, "
                  "the highest of any group.")
    lines.append("   -> Recommendation: Build a structured onboarding/retention "
                  "journey for the first 12 months (proactive check-ins, "
                  "early-tenure discounts, satisfaction surveys).\n")

    lines.append(f"3. SERVICE ENGAGEMENT (add-ons) reduces churn.")
    lines.append(f"   Customers with 0 add-on services churn at {addon_churn.iloc[0]*100:.1f}%, "
                  f"while those with the most add-ons churn at {addon_churn.iloc[-1]*100:.1f}%.")
    lines.append("   -> Recommendation: Bundle Online Security / Tech Support "
                  "into a free trial for at-risk, low-engagement customers to "
                  "increase stickiness.\n")

    lines.append(f"4. Top 5 predictive features from the model: {', '.join(top_drivers)}")
    lines.append("   -> Recommendation: Prioritize a targeted retention campaign "
                  "for customers who score high-risk on these features — model "
                  "output can feed directly into a CRM 'at-risk' flag.\n")

    lines.append("5. ESTIMATED BUSINESS IMPACT")
    total_customers = len(df)
    churners = df["Churn"].sum()
    lines.append(f"   Out of {total_customers:,} customers, {churners:,} churned "
                  f"({churners/total_customers:.1%}). Even a modest 10% reduction in "
                  f"churn among high-risk, flagged customers would retain roughly "
                  f"{int(churners * 0.10):,} additional customers.")

    return "\n".join(lines)


def main():
    df = get_processed_data()

    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "avg_monthly_spend", "num_addon_services",
    ]
    categorical_features = [c for c in df.columns
                             if c not in numeric_features + ["Churn", "is_low_engagement"]]

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    # --- Baseline: Logistic Regression ---
    log_reg = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        numeric_features, categorical_features,
    )
    log_reg.fit(X_train, y_train)
    lr_metrics, lr_pred = evaluate("Logistic Regression", log_reg, X_test, y_test)
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", "07_confusion_matrix_logreg.png")
    results.append(lr_metrics)

    # --- Random Forest ---
    rf = build_pipeline(
        RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        numeric_features, categorical_features,
    )
    rf.fit(X_train, y_train)
    rf_metrics, rf_pred = evaluate("Random Forest", rf, X_test, y_test)
    plot_confusion_matrix(y_test, rf_pred, "Random Forest", "08_confusion_matrix_rf.png")
    results.append(rf_metrics)

    # Pick best model by ROC-AUC (good metric for imbalanced churn data)
    best = max(results, key=lambda r: r["roc_auc"])
    best_pipeline = rf if best["model"] == "Random Forest" else log_reg
    print(f"\nBest model by ROC-AUC: {best['model']} ({best['roc_auc']})")

    # Save best model
    joblib.dump(best_pipeline, f"{MODEL_DIR}/churn_model.pkl")
    print(f"Saved best model to {MODEL_DIR}/churn_model.pkl")

    # Save metrics
    with open(f"{OUTPUT_DIR}/model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Feature importance (only meaningful for the RF model)
    imp_df = plot_feature_importance(rf, numeric_features, categorical_features)
    imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

    # Business recommendations
    recommendations = generate_business_recommendations(imp_df, df)
    with open(f"{OUTPUT_DIR}/business_recommendations.txt", "w") as f:
        f.write(recommendations)
    print("\n" + recommendations)


if __name__ == "__main__":
    main()
