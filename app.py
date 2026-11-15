import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from fpdf import FPDF
from scipy.ndimage import gaussian_filter
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QuorumQuench AI", page_icon="🧬", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f4f6f9; }
    h1, h2, h3 { color: #0b3d91; }
    .metric-card { background: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px; border: 1px solid #e0e0e0; }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #0b3d91; }
    .metric-label { font-size: 1rem; color: #6c757d; }
    .stButton>button { width: 100%; height: 3em; font-size: 18px; font-weight: bold; background: linear-gradient(90deg, #1f77b4, #00d4ff); color: white; border-radius: 8px; border: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; overflow-x: auto; flex-wrap: nowrap; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #e9ecef; border-radius: 4px 4px 0 0; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; font-weight: bold; color: #0b3d91; border-bottom: 3px solid #0b3d91; }
    .explain-box { background-color: #ffffff; border-left: 5px solid #0b3d91; padding: 15px; margin-top: 15px; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); font-size: 14px; }
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

# --- TABS (10 TABS) ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "⚡ Recommendation", "🎮 Sandbox", "🧠 XAI", "🌐 3D Pathway", "📈 ODE Sim", 
    "🔬 Biopsy", "💊 PK Curve", "🧬 Mutation Matrix", "🌀 Digital Twin", "📊 Compare Patients"
])

# ==========================================
# TAB 1: RECOMMENDATION
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
        
        st.markdown("**Molecular Balance Scale:**")
        st.markdown(f"""<div class="explain-box"><b>How to read this:</b> The scale shows the battle in the tumor. If the right side (Gamma Shield) is heavier, the pathway is protective (do not give QQ). If the left side (F. nucleatum) is heavier, QQ will help.</div>""", unsafe_allow_html=True)
        
        bacteria_weight = f_nucleatum_load * 1.5
        host_weight = gamma_effective * 0.8
        tilt = (host_weight - bacteria_weight) / 2.0 
        
        fig_scale = go.Figure()
        fig_scale.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=0.4, line=dict(color="black", width=6))
        fig_scale.add_shape(type="line", x0=0.3, y0=0, x1=0.7, y1=0, line=dict(color="black", width=6))
        x_left = 0.5 - 0.3 * np.cos(tilt); y_left = 0.4 - 0.3 * np.sin(tilt)
        x_right = 0.5 + 0.3 * np.cos(tilt); y_right = 0.4 + 0.3 * np.sin(tilt)
        fig_scale.add_shape(type="line", x0=x_left, y0=y_left, x1=x_right, y1=y_right, line=dict(color="#0b3d91", width=6))
        fig_scale.add_trace(go.Scatter(x=[x_left], y=[y_left-0.05], mode='markers+text', marker=dict(size=40, color='orange'), text=[f"F. nucleatum<br>{f_nucleatum_load:.2f}"], textposition="bottom center", showlegend=False))
        fig_scale.add_trace(go.Scatter(x=[x_right], y=[y_right-0.05], mode='markers+text', marker=dict(size=40, color=color), text=[f"Gamma Shield<br>{gamma_effective:.2f}"], textposition="bottom center", showlegend=False))
        fig_scale.update_layout(xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(range=[-0.1, 0.8], showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_scale, use_container_width=True)

    with col2:
        st.subheader("Biological Reasoning")
        if outcome == "HARMFUL": st.info("**High Effective Gamma:** The PI3K/AKT pathway is acting as a protective shield. QQ would remove this shield, increasing metastasis risk.")
        else: st.info("**Low Effective Gamma:** The PI3K/AKT pathway is driving metastasis. QQ will block the AI-2 signal and reduce tumor burden.")
            
        pdf_bytes = generate_pdf(sex, pik3ca, gamma_effective, f_nucleatum_load, outcome, risk_pct)
        st.download_button("📄 Download Clinical PDF Report", data=pdf_bytes, file_name="QQ_Clinical_Report.pdf", mime="application/pdf")
        
        if st.button("💾 Save to Patient History"):
            st.session_state.history.append({"ID": len(st.session_state.history)+1, "Sex": sex, "PIK3CA": pik3ca, "Gamma": round(gamma_effective, 2), "F. nucleatum": round(f_nucleatum_load, 2), "Outcome": outcome, "Risk": f"{risk_pct}%"})
            st.success("Saved to history!")

# ==========================================
# TAB 2: TREATMENT SANDBOX
# ==========================================
with tab2:
    st.subheader("🎮 Treatment Sandbox: Real-Time What-If Analysis")
    st.markdown("""<div class="explain-box"><b>How to use:</b> Adjust the sandbox sliders to simulate hypothetical patients. Watch how the Metastasis Trajectory curve changes. Up = Harmful. Down = Protective.</div>""", unsafe_allow_html=True)
    
    sandbox_qq = st.slider("Adjust QQ Dose (Sandbox)", 0.0, 1.5, qq_dose, 0.1)
    sandbox_gamma = st.slider("Adjust Gamma Level (Sandbox)", 0.1, 2.0, gamma_effective, 0.1)
    
    time_arr = np.linspace(0, 100, 100)
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    
    if sandbox_gamma > threshold:
        sandbox_metastasis = control_metastasis * (1 + 0.5 * sandbox_qq * (sandbox_gamma - threshold))
        sandbox_color = "#ff0033"
    else:
        sandbox_metastasis = control_metastasis * (1 - 0.8 * sandbox_qq)
        sandbox_color = "#00d4ff"

    fig_sandbox = go.Figure()
    fig_sandbox.add_trace(go.Scatter(x=time_arr, y=control_metastasis, mode='lines', name='Metastasis (No QQ)', line=dict(color='gray', dash='dash')))
    fig_sandbox.add_trace(go.Scatter(x=time_arr, y=sandbox_metastasis, mode='lines', name='Metastasis (QQ Treated)', line=dict(color=sandbox_color, width=4)))
    fig_sandbox.update_layout(title="Real-Time Metastasis Trajectory", template="plotly_white", height=400)
    st.plotly_chart(fig_sandbox, use_container_width=True)
    
    if sandbox_gamma > threshold:
        st.error(f"⚠️ **Result:** At Gamma = {sandbox_gamma:.2f}, QQ therapy is **HARMFUL** because it removes the protective pathway.")
    else:
        st.success(f"✅ **Result:** At Gamma = {sandbox_gamma:.2f}, QQ therapy is **PROTECTIVE** because it blocks the pro-metastatic signal.")

# ==========================================
# TAB 3: XAI
# ==========================================
with tab3:
    st.subheader("🧠 AI Explainability (XAI) Brain Map")
    st.markdown("""<div class="explain-box"><b>What is this?</b> A Waterfall plot that explains the "Brain" of the AI. Positive values (red) push the patient toward an adverse outcome. Negative values (green) push toward a protective outcome.</div>""", unsafe_allow_html=True)
    
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
# TAB 4: 3D PATHWAY
# ==========================================
with tab4:
    st.subheader("🌐 3D Interactive Biological Pathway")
    st.markdown("""<div class="explain-box"><b>Step-by-Step Activation:</b> <br>1. F. nucleatum adheres to the colon. <br>2. Virulence factors are injected. <br>3. PI3K is activated. <br>4. AKT (Gamma) creates the protective shield. <br>5. Metastasis is the outcome. Rotate the graph to see the spatial relationship.</div>""", unsafe_allow_html=True)
    
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
# TAB 5: 2D SIMULATION
# ==========================================
with tab5:
    st.subheader("📈 Advanced ODE Simulation: Host-Microbe Dynamics")
    st.markdown("""<div class="explain-box"><b>What is happening here?</b> The Top Graph shows bacterial growth (orange) and QQ suppression (purple). The Bottom Graph shows the downstream effect on metastasis. If Gamma is high, QQ causes the metastasis curve to spike (red line) because the protective shield is removed.</div>""", unsafe_allow_html=True)
    
    time_arr = np.linspace(0, 100, 100)
    b_growth = 1 / (1 + np.exp(-0.1 * (time_arr - 30))) * f_nucleatum_load
    b_qq = b_growth * (1 - 0.9 * qq_dose)
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    if outcome == "PROTECTIVE": qq_metastasis = control_metastasis * (1 - 0.8 * qq_dose)
    else: qq_metastasis = control_metastasis * (1 + 0.5 * qq_dose * (gamma_effective - threshold))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_arr, y=b_growth, mode='lines', name='F. nucleatum (No QQ)', line=dict(color='orange', dash='dash')))
    fig1.add_trace(go.Scatter(x=time_arr, y=b_qq, mode='lines', name='F. nucleatum (QQ Treated)', line=dict(color='purple', width=3)))
    fig1.update_layout(title="Bacterial Load Dynamics", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_arr, y=control_metastasis, mode='lines', name='Metastasis (No QQ)', line=dict(color='gray', dash='dash')))
    fig2.add_trace(go.Scatter(x=time_arr, y=qq_metastasis, mode='lines', name='Metastasis (QQ Treated)', line=dict(color=color, width=3)))
    fig2.update_layout(title="Metastasis Burden", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 6: VIRTUAL BIOPSY (Upgraded)
# ==========================================
with tab6:
    st.subheader("🔬 Virtual Biopsy: Histology Simulation")
    st.markdown("""<div class="explain-box"><b>What is this?</b> A simulated H&E stain of the colon tissue. Purple dots = <i>F. nucleatum</i> bacteria. Red/chaotic areas = tumor tissue. Use the sidebar to change the bacterial load and Gamma score to see the tissue change.</div>""", unsafe_allow_html=True)
    
    img_size = 400
    tissue = np.random.normal(0.3, 0.05, (img_size, img_size))
    
    # Add bacteria (purple dots)
    num_bacteria = int(f_nucleatum_load * 800)
    for _ in range(num_bacteria):
        x, y = np.random.randint(0, img_size, 2)
        tissue[max(0, x-3):min(img_size, x+3), max(0, y-3):min(img_size, y+3)] += 0.5
    
    # Add tumor chaos (red areas)
    tumor_intensity = gamma_effective * 0.2
    chaos = np.random.normal(0, tumor_intensity, (img_size, img_size))
    tissue += gaussian_filter(chaos, sigma=5) # Smooth the chaos into blobs
    
    fig_biopsy, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(tissue, cmap='RdPu', interpolation='bilinear') # Red-Purple colormap
    ax.axis('off')
    st.pyplot(fig_biopsy)

# ==========================================
# TAB 7: PK CURVE (Upgraded)
# ==========================================
with tab7:
    st.subheader("💊 Pharmacokinetics (PK) Curve")
    st.markdown("""<div class="explain-box"><b>What is this?</b> Simulates the concentration of QQ drug in the blood over 72 hours, including a second dose at 48 hours. The green line shows the drug concentration; the red dashed line shows the Minimum Inhibitory Concentration (MIC) required to suppress bacteria.</div>""", unsafe_allow_html=True)
    
    time_pk = np.linspace(0, 72, 200)
    dose_1 = qq_dose * 1.5
    conc_1 = dose_1 * np.exp(-0.05 * time_pk) * (1 - np.exp(-2.0 * time_pk)) # First dose
    conc_2 = dose_1 * np.exp(-0.05 * (time_pk - 48)) * (1 - np.exp(-2.0 * (time_pk - 48))) * (time_pk >= 48) # Second dose
    total_conc = conc_1 + conc_2
    
    fig_pk = go.Figure()
    fig_pk.add_trace(go.Scatter(x=time_pk, y=total_conc, mode='lines', name='Drug Concentration', line=dict(color='#00d4ff', width=4)))
    fig_pk.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="MIC (Minimum Inhibitory Conc.)")
    fig_pk.update_layout(title="Plasma Concentration of SsoPox (with 2nd Dose at 48h)", xaxis_title="Time (Hours)", yaxis_title="Concentration (mg/L)", template="plotly_white", height=400)
    st.plotly_chart(fig_pk, use_container_width=True)

# ==========================================
# TAB 8: MUTATION MATRIX
# ==========================================
with tab8:
    st.subheader("🧬 Mutation Impact Matrix")
    st.markdown("""<div class="explain-box"><b>What is this?</b> A heatmap showing how different mutations interact to change the risk score. Dark red = high risk of QQ harm. Dark green = QQ is protective.</div>""", unsafe_allow_html=True)
    
    mutations = ['Wildtype', 'PIK3CA', 'PTEN', 'KRAS', 'PIK3CA+PTEN']
    risks = [30, 65, 45, 70, 85]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=[risks], x=mutations, y=['Risk Score'],
        colorscale='RdYlGn_r',
        text=[f"{r}%" for r in risks], texttemplate="%{text}",
        colorbar=dict(title="Risk %")
    ))
    fig_heatmap.update_layout(title="Risk of QQ Harm by Mutation Type", template="plotly_white", height=300)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ==========================================
# TAB 9: DIGITAL TWIN (Upgraded)
# ==========================================
with tab9:
    st.subheader("🌀 Digital Twin: Monte Carlo Simulation")
    st.markdown("""<div class="explain-box"><b>What is this?</b> 100 parallel simulations showing the uncertainty of the patient's future. The thick line is the average. Faint lines are individual possible futures.</div>""", unsafe_allow_html=True)
    
    time_arr = np.linspace(0, 100, 100)
    fig_mc = go.Figure()
    
    for i in range(100):
        noise = np.random.normal(1.0, 0.15, 100)
        if outcome == "PROTECTIVE":
            met = (1 / (1 + np.exp(-0.1 * (time_arr - 50)))) * (1 - 0.8 * qq_dose) * noise
            line_color = 'rgba(0, 212, 255, 0.1)'
        else:
            met = (1 / (1 + np.exp(-0.1 * (time_arr - 50)))) * (1 + 0.5 * qq_dose * (gamma_effective - threshold)) * noise
            line_color = 'rgba(255, 0, 51, 0.1)'
        fig_mc.add_trace(go.Scatter(x=time_arr, y=met, mode='lines', line=dict(color=line_color, width=1), showlegend=False))
    
    if outcome == "PROTECTIVE":
        avg_met = (1 / (1 + np.exp(-0.1 * (time_arr - 50)))) * (1 - 0.8 * qq_dose)
        main_color = "#00d4ff"
    else:
        avg_met = (1 / (1 + np.exp(-0.1 * (time_arr - 50)))) * (1 + 0.5 * qq_dose * (gamma_effective - threshold))
        main_color = "#ff0033"
    fig_mc.add_trace(go.Scatter(x=time_arr, y=avg_met, mode='lines', name='Average', line=dict(color=main_color, width=4)))
    fig_mc.update_layout(title="100 Simulated Metastasis Trajectories", template="plotly_white", height=500)
    st.plotly_chart(fig_mc, use_container_width=True)

# ==========================================
# TAB 10: COMPARE PATIENTS (Restored)
# ==========================================
with tab10:
    st.subheader("📊 Clinical Trial Mode: Side-by-Side Comparison")
    st.markdown("""<div class="explain-box"><b>How to use:</b> Save at least 2 patients in Tab 1. This tab will generate a Scatter Plot showing how the patients cluster based on their Gamma and F. nucleatum load.</div>""", unsafe_allow_html=True)
    
    if len(st.session_state.history) < 2:
        st.info("Please save at least 2 patients to the history (Tab 1) to compare them.")
    else:
        df = pd.DataFrame(st.session_state.history)
        
        fig_compare = go.Figure()
        for i, row in df.iterrows():
            fig_compare.add_trace(go.Scatter(
                x=[row['Gamma']], y=[row['F. nucleatum']],
                mode='markers+text',
                marker=dict(size=25, color='#ff0033' if row['Outcome'] == 'HARMFUL' else '#00d4ff', line=dict(color='black', width=2)),
                text=[f"Patient {row['ID']}"],
                textposition="top center",
                name=f"Patient {row['ID']} ({row['Outcome']})"
            ))
            
        fig_compare.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_compare.update_layout(
            title="Patient Clustering: Gamma vs F. nucleatum Load",
            xaxis_title="Effective Gamma",
            yaxis_title="F. nucleatum Load",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        st.markdown("### Cohort Summary Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Patients Saved", len(df))
        c2.metric("Harmful Outcomes", len(df[df['Outcome'] == 'HARMFUL']))
        c3.metric("Protective Outcomes", len(df[df['Outcome'] == 'PROTECTIVE']))
        
        st.dataframe(df, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI | Omniscient Version | Developed by Muhammad Imad Khan Khalil")
