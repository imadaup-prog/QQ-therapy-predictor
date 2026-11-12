import streamlit as st

st.set_page_config(page_title="QQ Therapy Predictor", page_icon="🧬")

st.title("🧬 Quorum Quenching (QQ) Therapy Predictor")
st.markdown("""
This tool uses our ODE model to predict whether a colorectal cancer (CRC) patient will benefit from **Quorum Quenching (QQ)** therapy based on their molecular profile.
""")

st.sidebar.header("Patient Parameters")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
gamma = st.sidebar.slider("Gamma (p-AKT / PI3K/AKT Activity)", 0.1, 2.0, 0.5, 0.1)
pik3ca = st.sidebar.selectbox("PIK3CA Mutation Status", ["Wildtype", "Mutated"])
s0 = st.sidebar.slider("S0 (Baseline Immune Suppression)", 0.1, 1.0, 0.5, 0.1)

# Model Logic
threshold = 1.0
if sex == "Female":
    threshold = 0.8
elif sex == "Male":
    threshold = 1.2
if pik3ca == "Mutated":
    threshold -= 0.1

st.header("Prediction Results")
st.markdown(f"**Calculated Gamma Threshold for this patient:** `{threshold:.1f}`")

if gamma > threshold:
    st.error("❌ **DO NOT GIVE QQ THERAPY**")
    st.write(f"High Gamma ({gamma}) indicates a protective pathway. QQ would remove this shield, increasing risk.")
else:
    st.success("✅ **GIVE QQ THERAPY**")
    st.write(f"Low Gamma ({gamma}) indicates a pro-metastatic pathway. QQ will block this signal.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model by:** Muhammad Imad Khan Khalil")
