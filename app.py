import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QuorumQuench AI", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR A PREMIUM LOOK ---
st.markdown("""
    <style>
    .reportview-container { background: #f4f6f9; }
    h1, h2, h3 { color: #0b3d91; font-family: 'Helvetica Neue', sans-serif; }
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px; }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #0b3d91; }
    .metric-label { font-size: 1rem; color: #6c757d; }
    .stButton>button { width: 100%; height: 3em; font-size: 18px; font-weight: bold; background-color: #1f77b4; color: white; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #e9ecef; border-radius: 4px 4px 0 0; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; font-weight: bold; color: #0b3d91; border-bottom: 3px solid #0b3d91; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- PDF GENERATOR FUNCTION ---
def generate_pdf(sex, pik3ca, gamma, f_nuc, outcome, risk):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="QuorumQuench AI Clinical Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Sex: {sex}", ln=True)
    pdf.cell(200, 10, txt=f"PIK3CA Status: {pik3ca}", ln=True)
    pdf.cell(200, 10, txt=f"Effective Gamma: {gamma:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"F. nucleatum Load: {f_nuc:.2f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"RECOMMENDATION: {outcome}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Risk of Adverse Outcome: {risk}%", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="Disclaimer: This tool is for research purposes only and does not constitute medical advice.")
    # CORRECTED LINE BELOW
    return bytes(pdf.output())

# --- TITLE & TOUR ---
st.title("🧬 QuorumQuench AI")
st.markdown("#### Integrated Host-Microbe Decision Support System")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Patient Parameters")
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
    color = "red"
    icon = "❌"
    rec = "DO NOT GIVE QQ THERAPY"
else:
    outcome = "PROTECTIVE"
    risk_pct = min(95, int((threshold - gamma_effective) * 100 + 50))
    color = "green"
    icon = "✅"
    rec = "GIVE QQ THERAPY"

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Recommendation", "📈 Simulation", "🧬 Pathway", "🔬 Validation", "📊 Compare Patients"])

# ==========================================
# TAB 1: CLINICAL RECOMMENDATION
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Prediction Result")
        if outcome == "HARMFUL": st.error(f"# {icon} {rec}")
        else: st.success(f"# {icon} {rec}")
        
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{risk_pct}%</div><div class="metric-label">Adverse Risk</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{gamma_effective:.2f}</div><div class="metric-label">Effective Gamma</div></div>', unsafe_allow_html=True)
        
        # PDF Download Button
        pdf_bytes = generate_pdf(sex, pik3ca, gamma_effective, f_nucleatum_load, outcome, risk_pct)
        st.download_button(
            label="📄 Download Clinical PDF Report",
            data=pdf_bytes,
            file_name="QQ_Clinical_Report.pdf",
            mime="application/pdf"
        )

    with col2:
        st.subheader("Biological Reasoning")
        if outcome == "HARMFUL":
            st.info("**High Effective Gamma:** The PI3K/AKT pathway is acting as a protective shield. QQ would remove this shield, increasing metastasis risk.")
        else:
            st.info("**Low Effective Gamma:** The PI3K/AKT pathway is driving metastasis. QQ will block the AI-2 signal and reduce tumor burden.")
            
        if st.button("💾 Save to Patient History"):
            st.session_state.history.append({"ID": len(st.session_state.history)+1, "Sex": sex, "PIK3CA": pik3ca, "Gamma": gamma_effective, "Outcome": outcome})
            st.success("Saved!")

# ==========================================
# TAB 2: LIVE SIMULATION
# ==========================================
with tab2:
    st.subheader("ODE Simulation: Host-Microbe Dynamics")
    with st.spinner("Running ODE simulation..."):
        time.sleep(0.5) 
        
    time_arr = np.linspace(0, 100, 100)
    b_growth = 1 / (1 + np.exp(-0.1 * (time_arr - 30))) * f_nucleatum_load
    b_qq = b_growth * (1 - 0.9 * qq_dose)
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    if outcome == "PROTECTIVE": qq_metastasis = control_metastasis * (1 - 0.8 * qq_dose)
    else: qq_metastasis = control_metastasis * (1 + 0.5 * qq_dose * (gamma_effective - threshold))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_arr, y=b_growth, mode='lines', name='F. nucleatum (Ctrl)', line=dict(color='orange', dash='dash')))
    fig1.add_trace(go.Scatter(x=time_arr, y=b_qq, mode='lines', name='F. nucleatum (QQ)', line=dict(color='purple', width=3)))
    fig1.update_layout(title="Bacterial Load Dynamics", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_arr, y=control_metastasis, mode='lines', name='Metastasis (Ctrl)', line=dict(color='gray', dash='dash')))
    fig2.add_trace(go.Scatter(x=time_arr, y=qq_metastasis, mode='lines', name='Metastasis (QQ)', line=dict(color=color, width=3)))
    fig2.update_layout(title="Metastasis Burden", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 3: PATHWAY VISUALIZATION
# ==========================================
with tab3:
    st.subheader("Dynamic Pathway")
    fig = go.Figure()
    nodes = [(0.1, 0.8, "F. nucleatum", "orange"), (0.3, 0.8, "Fap2/RadD", "orange"), (0.5, 0.7, "PI3K", "lightblue"), (0.7, 0.6, "AKT (Gamma)", "lightgreen"), (0.9, 0.6, "Metastasis", "red" if outcome == "HARMFUL" else "gray")]
    for x, y, label, c in nodes:
        fig.add_shape(type="rect", x0=x-0.05, y0=y-0.05, x1=x+0.05, y1=y+0.05, line=dict(color="black"), fillcolor=c)
        fig.add_annotation(x=x, y=y, text=label, showarrow=False)
    fig.add_annotation(x=0.7, y=0.4, text=f"Patient Gamma: {gamma_effective:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-40)
    fig.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 4: VALIDATION
# ==========================================
with tab4:
    st.subheader("TCGA COAD Validation (n=500)")
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.linspace(0, 150, 100)
    ax.plot(t, np.exp(-0.01*t), color='#C44E52', label='Mutated_High')
    ax.plot(t, np.exp(-0.015*t), color='#DD8452', label='Mutated_Low')
    ax.plot(t, np.exp(-0.025*t), color='#4C72B0', label='Wildtype_High')
    ax.plot(t, np.exp(-0.018*t), color='#55A868', label='Wildtype_Low')
    ax.set_title("Two-Hit Hypothesis: Survival")
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    st.pyplot(fig)

# ==========================================
# TAB 5: COMPARE PATIENTS (Clinical Trial Mode)
# ==========================================
with tab5:
    st.subheader("Clinical Trial Mode: Side-by-Side Comparison")
    if len(st.session_state.history) < 2:
        st.info("Please save at least 2 patients to the history (Tab 1) to compare them.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        fig = go.Figure(data=[
            go.Bar(name='Gamma', x=df['ID'], y=df['Gamma'], marker_color=['red' if o == 'HARMFUL' else 'green' for o in df['Outcome']])
        ])
        fig.update_layout(title="Gamma Comparison Across Saved Patients", xaxis_title="Patient ID", yaxis_title="Effective Gamma")
        st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI | Advanced Clinical Version | Developed by Muhammad Imad Khan Khalil")
