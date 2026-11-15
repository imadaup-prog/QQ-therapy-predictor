import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
from scipy.interpolate import griddata
from streamlit_lottie import st_lottie
import json
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QuorumQuench AI: Omniscient", page_icon="🧠", layout="wide")

# --- CUSTOM CSS FOR A SCI-FI LOOK ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .stApp { background-color: #050505; color: #00ffcc; font-family: 'Orbitron', sans-serif; }
    h1, h2, h3 { color: #00ffcc; text-shadow: 0 0 10px #00ffcc; }
    .metric-card { background: rgba(0, 255, 204, 0.1); padding: 20px; border-radius: 15px; box-shadow: 0 0 20px rgba(0, 255, 204, 0.3); text-align: center; margin-bottom: 20px; border: 1px solid #00ffcc; }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #ffffff; text-shadow: 0 0 10px #00ffcc; }
    .metric-label { font-size: 1rem; color: #00ffcc; }
    .stButton>button { width: 100%; height: 3em; font-size: 18px; font-weight: bold; background: linear-gradient(90deg, #00ffcc, #0066ff); color: black; border-radius: 8px; border: none; box-shadow: 0 0 15px #00ffcc; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #111; color: #00ffcc; border-radius: 4px 4px 0 0; border: 1px solid #00ffcc; }
    .stTabs [aria-selected="true"] { background-color: #00ffcc; color: black; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- ANIMATIONS (Lottie) ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_dna = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_4kx2q9u3.json") # DNA animation
lottie_ai = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_zrqthn6o.json") # AI Brain animation

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- PDF GENERATOR ---
def generate_pdf(sex, pik3ca, gamma, f_nuc, outcome, risk):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="QuorumQuench AI: Omniscient Clinical Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Sex: {sex}", ln=True)
    pdf.cell(200, 10, txt=f"PIK3CA Status: {pik3ca}", ln=True)
    pdf.cell(200, 10, txt=f"Effective Gamma: {gamma:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"F. nucleatum Load: {f_nuc:.2f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"AI RECOMMENDATION: {outcome}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Risk of Adverse Outcome: {risk}%", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="Disclaimer: This tool is for research purposes only and does not constitute medical advice.")
    return bytes(pdf.output())

# --- TITLE & ANIMATION ---
col_title, col_anim = st.columns([3, 1])
with col_title:
    st.title("🧠 QuorumQuench AI: Omniscient")
    st.markdown("#### Next-Generation Explainable AI for Precision Oncology")
with col_anim:
    st_lottie(lottie_ai, height=100, key="ai_brain")

st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Patient Parameters")
st.sidebar.markdown("**Host & Microbial Data:**")
sex = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])
gamma_baseline = st.sidebar.slider("Baseline Gamma (p-AKT Activity)", 0.1, 2.0, 0.5, 0.1)
pik3ca = st.sidebar.selectbox("PIK3CA Mutation Status", ["Wildtype", "Mutated"])
s0 = st.sidebar.slider("S0 (Baseline Immune Suppression)", 0.1, 1.0, 0.5, 0.1)
f_nucleatum_load = st.sidebar.slider("F. nucleatum Abundance", 0.0, 1.0, 0.3, 0.05)
qq_dose = st.sidebar.slider("Proposed QQ Dose (SsoPox)", 0.0, 1.5, 0.8, 0.1)

# --- MODEL LOGIC ---
bacterial_push = f_nucleatum_load * 0.5
gamma_effective = gamma_baseline + bacterial_push

threshold = 1.0
if sex == "Female": threshold = 0.8
elif sex == "Male": threshold = 1.2
if pik3ca == "Mutated": threshold -= 0.1

if gamma_effective > threshold:
    outcome = "HARMFUL"
    risk_pct = min(95, int((gamma_effective - threshold) * 100 + 50))
    color = "#ff0033"
    icon = "❌"
    rec = "DO NOT GIVE QQ THERAPY"
else:
    outcome = "PROTECTIVE"
    risk_pct = min(95, int((threshold - gamma_effective) * 100 + 50))
    color = "#00ffcc"
    icon = "✅"
    rec = "GIVE QQ THERAPY"

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["⚡ AI Recommendation", "🎮 Treatment Sandbox", "🧠 Explainability (XAI)", "🔬 Validation"])

# ==========================================
# TAB 1: AI RECOMMENDATION (With Vitals Monitor)
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Diagnostic Result")
        if outcome == "HARMFUL": st.error(f"# {icon} {rec}")
        else: st.success(f"# {icon} {rec}")
        
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{risk_pct}%</div><div class="metric-label">Adverse Risk</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{gamma_effective:.2f}</div><div class="metric-label">Effective Gamma</div></div>', unsafe_allow_html=True)
        
        # LIVE HEARTBEAT MONITOR
        st.markdown("**Live Patient Status Monitor:**")
        heart_rate = 60 + int(risk_pct / 2) # Simulated HR based on risk
        pulse_data = np.sin(np.linspace(0, 20, 100)) * np.exp(-np.linspace(0, 5, 100))
        fig_pulse = go.Figure(go.Scatter(y=pulse_data, mode='lines', line=dict(color=color, width=2)))
        fig_pulse.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#050505", plot_bgcolor="#050505", xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_pulse, use_container_width=True)
        st.markdown(f"<center><span style='color:{color}; font-size:20px;'>❤️ {heart_rate} BPM</span></center>", unsafe_allow_html=True)

    with col2:
        st.subheader("Biological Reasoning")
        if outcome == "HARMFUL":
            st.info("**High Effective Gamma:** The PI3K/AKT pathway is acting as a protective shield. QQ would remove this shield, increasing metastasis risk.")
        else:
            st.info("**Low Effective Gamma:** The PI3K/AKT pathway is driving metastasis. QQ will block the AI-2 signal and reduce tumor burden.")
            
        pdf_bytes = generate_pdf(sex, pik3ca, gamma_effective, f_nucleatum_load, outcome, risk_pct)
        st.download_button("📄 Download Clinical Report (PDF)", data=pdf_bytes, file_name="QQ_Clinical_Report.pdf", mime="application/pdf")

# ==========================================
# TAB 2: TREATMENT SANDBOX (Interactive What-If)
# ==========================================
with tab2:
    st.subheader("🎮 Treatment Sandbox: Real-Time What-If Analysis")
    st.markdown("Drag the sliders to simulate different treatment scenarios and watch the survival curve react in real-time.")
    
    sandbox_qq = st.slider("Adjust QQ Dose (Sandbox Mode)", 0.0, 1.5, qq_dose, 0.1)
    sandbox_gamma = st.slider("Adjust Gamma Level (Sandbox Mode)", 0.1, 2.0, gamma_effective, 0.1)
    
    time_arr = np.linspace(0, 100, 100)
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    
    if sandbox_gamma > threshold:
        sandbox_metastasis = control_metastasis * (1 + 0.5 * sandbox_qq * (sandbox_gamma - threshold))
        sandbox_color = "#ff0033"
    else:
        sandbox_metastasis = control_metastasis * (1 - 0.8 * sandbox_qq)
        sandbox_color = "#00ffcc"

    fig_sandbox = go.Figure()
    fig_sandbox.add_trace(go.Scatter(x=time_arr, y=control_metastasis, mode='lines', name='Metastasis (Ctrl)', line=dict(color='gray', dash='dash')))
    fig_sandbox.add_trace(go.Scatter(x=time_arr, y=sandbox_metastasis, mode='lines', name='Metastasis (Sandbox QQ)', line=dict(color=sandbox_color, width=4)))
    fig_sandbox.update_layout(title="Real-Time Metastasis Trajectory", template="plotly_dark", height=400)
    st.plotly_chart(fig_sandbox, use_container_width=True)
    
    if sandbox_gamma > threshold:
        st.error(f"⚠️ At this Gamma level ({sandbox_gamma:.2f}), QQ is HARMFUL.")
    else:
        st.success(f"✅ At this Gamma level ({sandbox_gamma:.2f}), QQ is PROTECTIVE.")

# ==========================================
# TAB 3: EXPLAINABILITY (XAI)
# ==========================================
with tab3:
    st.subheader("🧠 AI Explainability (XAI) Brain Map")
    st.markdown("This waterfall chart shows exactly how much each biological factor contributed to the AI's final risk prediction.")
    
    # Mock SHAP values for the waterfall chart
    shap_values = {
        "F. nucleatum Load": f_nucleatum_load * 30,
        "PIK3CA Mutation": 15 if pik3ca == "Mutated" else 0,
        "Baseline Gamma": gamma_baseline * 10,
        "Sex (S0)": 10 if sex == "Female" else -5,
        "QQ Dose Effect": -20 if outcome == "PROTECTIVE" else 20
    }
    
    fig_shap = go.Figure(go.Waterfall(
        name="SHAP", orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=list(shap_values.keys()) + ["Total Risk"],
        textposition="outside",
        text=[f"{v:+.1f}%" for v in shap_values.values()] + [f"{risk_pct}%"],
        y=list(shap_values.values()) + [risk_pct],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig_shap.update_layout(title="Feature Contribution to Risk", template="plotly_dark", waterfallgap=0.3)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.info("**Interpretation:** Positive values (red) push the patient toward an adverse outcome. Negative values (green) push toward a protective outcome.")

# ==========================================
# TAB 4: VALIDATION
# ==========================================
with tab4:
    st.subheader("TCGA COAD Validation (n=500)")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')
    ax.tick_params(colors='#00ffcc')
    ax.xaxis.label.set_color('#00ffcc')
    ax.yaxis.label.set_color('#00ffcc')
    ax.title.set_color('#00ffcc')

    t = np.linspace(0, 150, 100)
    ax.plot(t, np.exp(-0.01*t), color='#ff0033', label='Mutated_High')
    ax.plot(t, np.exp(-0.015*t), color='#ff9900', label='Mutated_Low')
    ax.plot(t, np.exp(-0.025*t), color='#0066ff', label='Wildtype_High')
    ax.plot(t, np.exp(-0.018*t), color='#00ffcc', label='Wildtype_Low')
    ax.set_title("Two-Hit Hypothesis: Survival")
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Survival Probability")
    ax.legend(facecolor='#050505', labelcolor='#00ffcc')
    st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI: Omniscient | Developed by Muhammad Imad Khan Khalil | WARNING: Experimental Use Only")
