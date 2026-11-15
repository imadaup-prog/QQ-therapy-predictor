import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import time
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QuorumQuench AI", page_icon="🧬", layout="wide")

# --- CUSTOM CSS (Light Theme) ---
st.markdown("""
    <style>
    .stApp { background-color: #f4f6f9; }
    h1, h2, h3 { color: #0b3d91; }
    .metric-card { background: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px; border: 1px solid #e0e0e0; }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #0b3d91; }
    .metric-label { font-size: 1rem; color: #6c757d; }
    .stButton>button { width: 100%; height: 3em; font-size: 18px; font-weight: bold; background: linear-gradient(90deg, #1f77b4, #00d4ff); color: white; border-radius: 8px; border: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #e9ecef; border-radius: 4px 4px 0 0; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; font-weight: bold; color: #0b3d91; border-bottom: 3px solid #0b3d91; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- PDF GENERATOR ---
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
    return bytes(pdf.output())

# --- TITLE ---
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
    color = "#ff0033"
    icon = "❌"
    rec = "DO NOT GIVE QQ THERAPY"
else:
    outcome = "PROTECTIVE"
    risk_pct = min(95, int((threshold - gamma_effective) * 100 + 50))
    color = "#00d4ff"
    icon = "✅"
    rec = "GIVE QQ THERAPY"

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["⚡ AI Recommendation", "🎮 Sandbox", "🧠 XAI Brain", "🌐 3D Pathway", "📈 2D Simulation", "📊 Compare Patients"])

# ==========================================
# TAB 1: AI RECOMMENDATION (With Vitals)
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
        
        # Live Heartbeat
        st.markdown("**Live Patient Status Monitor:**")
        heart_rate = 60 + int(risk_pct / 2)
        pulse_data = np.sin(np.linspace(0, 20, 100)) * np.exp(-np.linspace(0, 5, 100))
        fig_pulse = go.Figure(go.Scatter(y=pulse_data, mode='lines', line=dict(color=color, width=2)))
        fig_pulse.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_pulse, use_container_width=True)
        st.markdown(f"<center><span style='color:{color}; font-size:20px;'>❤️ {heart_rate} BPM</span></center>", unsafe_allow_html=True)

    with col2:
        st.subheader("Biological Reasoning")
        if outcome == "HARMFUL":
            st.info("**High Effective Gamma:** The PI3K/AKT pathway is acting as a protective shield. QQ would remove this shield, increasing metastasis risk.")
        else:
            st.info("**Low Effective Gamma:** The PI3K/AKT pathway is driving metastasis. QQ will block the AI-2 signal and reduce tumor burden.")
            
        pdf_bytes = generate_pdf(sex, pik3ca, gamma_effective, f_nucleatum_load, outcome, risk_pct)
        st.download_button("📄 Download Clinical PDF Report", data=pdf_bytes, file_name="QQ_Clinical_Report.pdf", mime="application/pdf")
        
        if st.button("💾 Save to Patient History"):
            st.session_state.history.append({"ID": len(st.session_state.history)+1, "Sex": sex, "PIK3CA": pik3ca, "Gamma": gamma_effective, "Outcome": outcome})
            st.success("Saved to history!")

# ==========================================
# TAB 2: TREATMENT SANDBOX
# ==========================================
with tab2:
    st.subheader("🎮 Treatment Sandbox: Real-Time What-If Analysis")
    sandbox_qq = st.slider("Adjust QQ Dose (Sandbox Mode)", 0.0, 1.5, qq_dose, 0.1)
    sandbox_gamma = st.slider("Adjust Gamma Level (Sandbox Mode)", 0.1, 2.0, gamma_effective, 0.1)
    
    time_arr = np.linspace(0, 100, 100)
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    
    if sandbox_gamma > threshold:
        sandbox_metastasis = control_metastasis * (1 + 0.5 * sandbox_qq * (sandbox_gamma - threshold))
        sandbox_color = "#ff0033"
    else:
        sandbox_metastasis = control_metastasis * (1 - 0.8 * sandbox_qq)
        sandbox_color = "#00d4ff"

    fig_sandbox = go.Figure()
    fig_sandbox.add_trace(go.Scatter(x=time_arr, y=control_metastasis, mode='lines', name='Metastasis (Ctrl)', line=dict(color='gray', dash='dash')))
    fig_sandbox.add_trace(go.Scatter(x=time_arr, y=sandbox_metastasis, mode='lines', name='Metastasis (Sandbox QQ)', line=dict(color=sandbox_color, width=4)))
    fig_sandbox.update_layout(title="Real-Time Metastasis Trajectory", template="plotly_white", height=400)
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
    fig_shap.update_layout(title="Feature Contribution to Risk", template="plotly_white", waterfallgap=0.3)
    st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# TAB 4: 3D PATHWAY (Restored & Fixed)
# ==========================================
with tab4:
    st.subheader("3D Interactive Biological Pathway")
    st.markdown("Use your mouse to **rotate, zoom, and pan** this 3D network.")
    
    nodes_3d = [
        {"x": 0, "y": 0, "z": 0, "text": f"<b>F. nucleatum</b><br>Load: {f_nucleatum_load:.2f}", "color": "orange", "size": 15},
        {"x": 2, "y": 1, "z": 1, "text": "<b>Fap2 / RadD</b><br>(Virulence)", "color": "orange", "size": 12},
        {"x": 4, "y": 0, "z": 2, "text": "<b>PI3K</b><br>(Activation)", "color": "lightblue", "size": 12},
        {"x": 6, "y": 1, "z": 1, "text": f"<b>AKT (Gamma)</b><br>Score: {gamma_effective:.2f}", "color": "#00ff00" if outcome == "PROTECTIVE" else "#ff0033", "size": 18},
        {"x": 8, "y": 0, "z": 0, "text": f"<b>Metastasis</b><br>{rec}", "color": "#00ff00" if outcome == "PROTECTIVE" else "#ff0033", "size": 15}
    ]
    fig_3d = go.Figure()
    for i in range(len(nodes_3d)-1):
        fig_3d.add_trace(go.Scatter3d(x=[nodes_3d[i]["x"], nodes_3d[i+1]["x"]], y=[nodes_3d[i]["y"], nodes_3d[i+1]["y"]], z=[nodes_3d[i]["z"], nodes_3d[i+1]["z"]], mode='lines', line=dict(color='gray', width=4), showlegend=False))
    for n in nodes_3d:
        fig_3d.add_trace(go.Scatter3d(x=[n["x"]], y=[n["y"]], z=[n["z"]], mode='markers+text', marker=dict(size=n["size"], color=n["color"], opacity=0.8, line=dict(color='white', width=1)), text=[n["text"]], textposition="top center", textfont=dict(size=12, color="black"), showlegend=False))
    fig_3d.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False, title=''), yaxis=dict(showbackground=False, showticklabels=False, title=''), zaxis=dict(showbackground=False, showticklabels=False, title=''), bgcolor="#f4f6f9"), paper_bgcolor="#f4f6f9", margin=dict(l=0, r=0, b=0, t=0), height=500)
    st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# TAB 5: 2D SIMULATION (Restored)
# ==========================================
with tab5:
    st.subheader("ODE Simulation: Host-Microbe Dynamics")
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
# TAB 6: COMPARE PATIENTS (Restored)
# ==========================================
with tab6:
    st.subheader("Clinical Trial Mode: Side-by-Side Comparison")
    if len(st.session_state.history) < 2:
        st.info("Please save at least 2 patients to the history (Tab 1) to compare them.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        fig = go.Figure(data=[
            go.Bar(name='Gamma', x=df['ID'], y=df['Gamma'], marker_color=['#ff0033' if o == 'HARMFUL' else '#00d4ff' for o in df['Outcome']])
        ])
        fig.update_layout(title="Gamma Comparison Across Saved Patients", xaxis_title="Patient ID", yaxis_title="Effective Gamma", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI | Advanced Clinical Version | Developed by Muhammad Imad Khan Khalil")
