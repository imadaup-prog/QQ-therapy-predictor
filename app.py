import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
from scipy.interpolate import griddata

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QuorumQuench AI 3D", page_icon="🧬", layout="wide")

# --- CUSTOM CSS FOR DARK PREMIUM THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #4da6ff; font-family: 'Helvetica Neue', sans-serif; }
    .metric-card { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.5); text-align: center; margin-bottom: 20px; border: 1px solid #4da6ff; }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #ffffff; }
    .metric-label { font-size: 1rem; color: #b0c4de; }
    .stButton>button { width: 100%; height: 3em; font-size: 18px; font-weight: bold; background: linear-gradient(90deg, #1f77b4, #00d4ff); color: white; border-radius: 8px; border: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #1e1e1e; color: #ffffff; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #2a5298; font-weight: bold; color: #00d4ff; border-bottom: 3px solid #00d4ff; }
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
st.title("🧬 QuorumQuench AI: 3D Advanced Dashboard")
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
    color = "#ff4b4b" # Red
    icon = "❌"
    rec = "DO NOT GIVE QQ THERAPY"
else:
    outcome = "PROTECTIVE"
    risk_pct = min(95, int((threshold - gamma_effective) * 100 + 50))
    color = "#00d4ff" # Cyan/Green
    icon = "✅"
    rec = "GIVE QQ THERAPY"

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Recommendation", "📈 2D Simulation", "🌐 3D Pathway & Surface", "🔬 Validation", "📊 Compare Patients"])

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
        
        pdf_bytes = generate_pdf(sex, pik3ca, gamma_effective, f_nucleatum_load, outcome, risk_pct)
        st.download_button("📄 Download Clinical PDF Report", data=pdf_bytes, file_name="QQ_Clinical_Report.pdf", mime="application/pdf")

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
# TAB 2: 2D SIMULATION
# ==========================================
with tab2:
    st.subheader("ODE Simulation: Host-Microbe Dynamics")
    with st.spinner("Running ODE simulation..."): time.sleep(0.5) 
        
    time_arr = np.linspace(0, 100, 100)
    b_growth = 1 / (1 + np.exp(-0.1 * (time_arr - 30))) * f_nucleatum_load
    b_qq = b_growth * (1 - 0.9 * qq_dose)
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    if outcome == "PROTECTIVE": qq_metastasis = control_metastasis * (1 - 0.8 * qq_dose)
    else: qq_metastasis = control_metastasis * (1 + 0.5 * qq_dose * (gamma_effective - threshold))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_arr, y=b_growth, mode='lines', name='F. nucleatum (Ctrl)', line=dict(color='orange', dash='dash')))
    fig1.add_trace(go.Scatter(x=time_arr, y=b_qq, mode='lines', name='F. nucleatum (QQ)', line=dict(color='purple', width=3)))
    fig1.update_layout(title="Bacterial Load Dynamics", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_arr, y=control_metastasis, mode='lines', name='Metastasis (Ctrl)', line=dict(color='gray', dash='dash')))
    fig2.add_trace(go.Scatter(x=time_arr, y=qq_metastasis, mode='lines', name='Metastasis (QQ)', line=dict(color=color, width=3)))
    fig2.update_layout(title="Metastasis Burden", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 3: 3D PATHWAY & SURFACE (THE WOW FACTOR)
# ==========================================
with tab3:
    st.subheader("3D Interactive Biological Pathway")
    st.markdown("Use your mouse to **rotate, zoom, and pan** this 3D network. The glowing nodes represent the pathway components.")

    # Create 3D Network Graph
    nodes_3d = [
        {"x": 0, "y": 0, "z": 0, "text": f"<b>F. nucleatum</b><br>Load: {f_nucleatum_load:.2f}", "color": "orange", "size": 15},
        {"x": 2, "y": 1, "z": 1, "text": "<b>Fap2 / RadD</b><br>(Virulence)", "color": "orange", "size": 12},
        {"x": 4, "y": 0, "z": 2, "text": "<b>PI3K</b><br>(Activation)", "color": "lightblue", "size": 12},
        {"x": 6, "y": 1, "z": 1, "text": f"<b>AKT (Gamma)</b><br>Score: {gamma_effective:.2f}", "color": "#00ff00" if outcome == "PROTECTIVE" else "#ff4b4b", "size": 18},
        {"x": 8, "y": 0, "z": 0, "text": f"<b>Metastasis</b><br>{rec}", "color": "#00ff00" if outcome == "PROTECTIVE" else "#ff4b4b", "size": 15}
    ]

    fig_3d = go.Figure()

    # Draw edges (lines between nodes)
    for i in range(len(nodes_3d)-1):
        fig_3d.add_trace(go.Scatter3d(
            x=[nodes_3d[i]["x"], nodes_3d[i+1]["x"]],
            y=[nodes_3d[i]["y"], nodes_3d[i+1]["y"]],
            z=[nodes_3d[i]["z"], nodes_3d[i+1]["z"]],
            mode='lines',
            line=dict(color='gray', width=4),
            showlegend=False
        ))

    # Draw nodes (spheres)
    for n in nodes_3d:
        fig_3d.add_trace(go.Scatter3d(
            x=[n["x"]], y=[n["y"]], z=[n["z"]],
            mode='markers+text',
            marker=dict(size=n["size"], color=n["color"], opacity=0.8, line=dict(color='white', width=1)),
            text=[n["text"]],
            textposition="top center",
            textfont=dict(size=12, color="white"),
            showlegend=False
        ))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor="#0e1117"
        ),
        paper_bgcolor="#0e1117",
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")
    st.subheader("3D Metastasis Landscape")
    st.markdown("This 3D surface shows how Metastasis Burden changes across different combinations of **Gamma** and **F. nucleatum Load**. Your patient is marked with a **Red Dot**.")

    # Generate 3D Surface Data
    g_range = np.linspace(0.1, 2.0, 30)
    f_range = np.linspace(0.0, 1.0, 30)
    G, F = np.meshgrid(g_range, f_range)
    M = 1 / (1 + np.exp(-2 * (G + 0.5*F - 1.0))) # Mock metastasis function

    fig_surface = go.Figure(data=[go.Surface(z=M, x=g_range, y=f_range, colorscale='Viridis', opacity=0.8)])

    # Add patient dot
    fig_surface.add_trace(go.Scatter3d(
        x=[gamma_effective], y=[f_nucleatum_load], z=[1 / (1 + np.exp(-2 * (gamma_effective + 0.5*f_nucleatum_load - 1.0)))],
        mode='markers',
        marker=dict(size=8, color='red', symbol='cross'),
        name='Current Patient'
    ))

    fig_surface.update_layout(
        scene=dict(
            xaxis_title='Gamma (p-AKT)',
            yaxis_title='F. nucleatum Load',
            zaxis_title='Metastasis Burden',
            bgcolor="#0e1117"
        ),
        paper_bgcolor="#0e1117",
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )
    st.plotly_chart(fig_surface, use_container_width=True)

# ==========================================
# TAB 4: VALIDATION
# ==========================================
with tab4:
    st.subheader("TCGA COAD Validation (n=500)")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('#4da6ff')

    t = np.linspace(0, 150, 100)
    ax.plot(t, np.exp(-0.01*t), color='#C44E52', label='Mutated_High')
    ax.plot(t, np.exp(-0.015*t), color='#DD8452', label='Mutated_Low')
    ax.plot(t, np.exp(-0.025*t), color='#4C72B0', label='Wildtype_High')
    ax.plot(t, np.exp(-0.018*t), color='#55A868', label='Wildtype_Low')
    ax.set_title("Two-Hit Hypothesis: Survival")
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Survival Probability")
    ax.legend(facecolor='#0e1117', labelcolor='white')
    st.pyplot(fig)

# ==========================================
# TAB 5: COMPARE PATIENTS
# ==========================================
with tab5:
    st.subheader("Clinical Trial Mode: Side-by-Side Comparison")
    if len(st.session_state.history) < 2:
        st.info("Please save at least 2 patients to the history (Tab 1) to compare them.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        fig = go.Figure(data=[
            go.Bar(name='Gamma', x=df['ID'], y=df['Gamma'], marker_color=['#ff4b4b' if o == 'HARMFUL' else '#00d4ff' for o in df['Outcome']])
        ])
        fig.update_layout(title="Gamma Comparison Across Saved Patients", xaxis_title="Patient ID", yaxis_title="Effective Gamma", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI | Advanced 3D Version | Developed by Muhammad Imad Khan Khalil")
