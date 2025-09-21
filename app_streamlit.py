# app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

MODEL_DIR = "models"
DATA_PATH = "yield_df.csv"

st.title("🌾 Crop Yield Prediction")
st.write("Enter features to predict yield (hg/ha). Models were trained on the Kaggle dataset.")  # noqa E501


# Load dataset (for dropdown lists)
@st.cache_data
def load_df():
    df = pd.read_csv(DATA_PATH)
    unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


df = load_df()

# get lists for Area and Item
if 'Area' in df.columns and 'Item' in df.columns:
    area_list = sorted(df['Area'].dropna().unique().tolist())
    item_list = sorted(df['Item'].dropna().unique().tolist())
else:
    st.error("Dataset must contain 'Area' and 'Item' columns to populate dropdowns.")  # noqa E501
    st.stop()

# available models (files present in models/)
model_files = {os.path.splitext(f)[0]: os.path.join(MODEL_DIR, f)
               for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')}
if not model_files:
    st.error("No model files found in models/. Run train.py first.")
    st.stop()

model_choice = st.selectbox("Choose model", list(model_files.keys()))

# Inputs
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Year", min_value=1900, max_value=2100, value=int(df['Year'].median()))  # noqa E501
    rainfall = st.number_input("Average rainfall (mm/year)", value=float(df['average_rain_fall_mm_per_year'].median()))   # noqa E501
    pesticides = st.number_input("Pesticides (tonnes)", value=float(df['pesticides_tonnes'].median()))  # noqa E501
with col2:
    avg_temp = st.number_input("Average temperature (°C)", value=float(df['avg_temp'].median()))   # noqa E501
    area = st.selectbox("Area (country/state)", area_list)
    item = st.selectbox("Crop (Item)", item_list)

if st.button("Predict"):
    # load pipeline
    model_path = model_files[model_choice]
    pipeline = joblib.load(model_path)

    # create input df
    input_df = pd.DataFrame([{
        'Year': year,
        'average_rain_fall_mm_per_year': rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': avg_temp,
        'Area': area,
        'Item': item
    }])

    # predict
    pred_hg = pipeline.predict(input_df)[0]  # in hectograms per hectare
    pred_kg = pred_hg / 10.0                 # 1 hg = 0.1 Kg
    pred_q = pred_kg / 100.0                 # 1 Quintal = 100 Kg

    # Farmer-friendly explanation 🌾
    st.subheader("🌾 Farmer Friendly Output")
    st.write(f"👉 With your inputs, the expected yield is about **{pred_kg:,.0f} Kg per hectare** (~ **{pred_q:,.1f} Quintals per hectare**).")  # noqa E501

    # Yield quality indicator (simple thresholds, adjust as per data)
    if pred_q > 30:
        st.success("🟢 This is a Good Yield!")
    elif pred_q > 15:
        st.warning("🟡 This is an Average Yield.")
    else:
        st.error("🔴 This is a Low Yield. Consider improving irrigation, fertilizer, or crop choice.")  # noqa E501

    # Technical details for experts
    st.subheader("🧑‍💻 Technical Output")
    st.write(f"Raw Prediction (hg/ha): {pred_hg:.2f}")

    if os.path.exists("outputs/metrics.csv"):
        metrics_df = pd.read_csv("outputs/metrics.csv")
        m = metrics_df[metrics_df['model'] == model_choice]
        if not m.empty:
            m = m.iloc[0]
            st.write("Model performance on test set:")
            st.write(f"- MAE: {m['mae']:.3f}")
            st.write(f"- RMSE: {m['rmse']:.3f}")
            st.write(f"- R²: {m['r2']:.3f}")
