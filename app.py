
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import joblib

from preprocess import engineer_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.title("📉 Customer Churn Prediction & Retention Tool")
    st.markdown(
        "Predicts the likelihood a telecom customer will churn, and "
        "surfaces a business recommendation based on their risk profile."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. Run `python src/train_model.py` from the "
            "project root first to train and save the model."
        )
        return

    model = load_model()

    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    with col2:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=float(monthly_charges * max(tenure, 1)), step=1.0
        )

    if st.button("Predict Churn Risk", type="primary"):
        input_df = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        input_df = engineer_features(input_df)

        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("Result")

        risk_pct = proba * 100
        if prediction == 1:
            st.error(f"⚠️ High Churn Risk — {risk_pct:.1f}% probability")
        else:
            st.success(f"✅ Low Churn Risk — {risk_pct:.1f}% probability")

        st.progress(min(proba, 1.0))

        st.subheader("Recommended Action")
        if risk_pct >= 60:
            st.markdown(
                "- **Immediate retention outreach** — this customer is highly likely to churn.\n"
                "- Offer a contract upgrade incentive (month-to-month → 1-year) with a loyalty discount.\n"
                "- Assign to a proactive customer success check-in within 7 days."
            )
        elif risk_pct >= 30:
            st.markdown(
                "- **Moderate risk** — monitor and consider a light-touch engagement offer.\n"
                "- Recommend an add-on service (Online Security / Tech Support) trial to increase stickiness."
            )
        else:
            st.markdown("- **Low risk** — no immediate action needed. Continue standard engagement.")

    st.divider()
    with st.expander("📊 About this model"):
        st.markdown(
            "Trained on the IBM Telco Customer Churn dataset (7,043 customers) "
            "using a Random Forest classifier. See the project README for full "
            "EDA, model comparison, and business recommendations."
        )


if __name__ == "__main__":
    main()
