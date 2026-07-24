
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from preprocess import get_processed_data

sns.set_style("whitegrid")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_churn_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df["Churn"].map({0: "Retained", 1: "Churned"}).value_counts().plot(
        kind="bar", ax=axes[0], color=["#2E86AB", "#E63946"]
    )
    axes[0].set_title("Overall Customer Churn Distribution")
    axes[0].set_ylabel("Number of Customers")
    axes[0].tick_params(axis="x", rotation=0)

    plot_df = df.copy()
    plot_df["Churn_label"] = plot_df["Churn"].map({0: "Retained", 1: "Churned"})
    sns.boxplot(data=plot_df, x="Churn_label", y="tenure", ax=axes[1],
                hue="Churn_label", palette=["#2E86AB", "#E63946"], legend=False)
    axes[1].set_title("Tenure Distribution: Retained vs Churned")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_churn_overview.png", dpi=150)
    plt.close()


def plot_churn_by_contract(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    churn_by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
    churn_by_contract.plot(kind="bar", color="#E63946")
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_churn_by_contract.png", dpi=150)
    plt.close()


def plot_churn_by_tenure_group(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    churn_by_tenure = df.groupby("tenure_group", observed=True)["Churn"].mean() * 100
    churn_by_tenure.plot(kind="bar", color="#F4A261")
    plt.title("Churn Rate by Tenure Group")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_churn_by_tenure_group.png", dpi=150)
    plt.close()


def plot_churn_by_monthly_charges(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True,
                common_norm=False, palette=["#2E86AB", "#E63946"])
    plt.title("Monthly Charges Distribution: Retained vs Churned")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_churn_by_monthly_charges.png", dpi=150)
    plt.close()


def plot_churn_by_addon_services(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    churn_by_addons = df.groupby("num_addon_services")["Churn"].mean() * 100
    churn_by_addons.plot(kind="bar", color="#457B9D")
    plt.title("Churn Rate by Number of Add-on Services Subscribed")
    plt.xlabel("Number of Add-on Services")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_churn_by_addon_services.png", dpi=150)
    plt.close()


def run_eda():
    df = get_processed_data()
    plot_churn_overview(df)
    plot_churn_by_contract(df)
    plot_churn_by_tenure_group(df)
    plot_churn_by_monthly_charges(df)
    plot_churn_by_addon_services(df)
    print(f"EDA charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_eda()
