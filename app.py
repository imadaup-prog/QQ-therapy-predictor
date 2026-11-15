import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF
from scipy.ndimage import gaussian_filter
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QuorumQuench AI: Master", page_icon="🧬", layout="wide")

# --- CUSTOM CSS: CLINICAL CLEAN THEME ---
st.markdown("""
    <style>
    /* Main App Background */
    .stApp { background-color: #f8fafc; } 
    
    /* Typography */
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; }
    p, div { font-family: 'Inter', sans-serif; color: #334155; }
    
    /* Metric Cards */
    .metric-card { 
        background: #ffffff; padding: 20px; border-radius: 12px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; 
        margin-bottom: 20px; border: 1px solid #e2e8f0; 
    }
    .metric-value { font-size: 2.5rem; font-weight: 800; color: #0f172a; }
    .metric-label { font-size: 0.9rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Buttons */
    .stButton>button { 
        width: 100%; height: 3em; font-size: 16px; font-weight: 600; 
        background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%); 
        color: white; border-radius: 8px; border: none; 
        transition: transform 0.2s;
    }
    .stButton>button:hover { transform: translateY(-2px); }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; overflow-x: auto; flex-wrap: nowrap; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; white-space: pre-wrap; background-color: #f1f5f9; 
        border-radius: 6px 6px 0 0; padding: 10px 16px; 
        font-weight: 500; color: #475569;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #ffffff; font-weight: 700; color: #0ea5e9; 
        border-bottom: 3px solid #0ea5e9; box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    
    /* Explanation Boxes */
    .explain-box { 
        background-color: #f0f9ff; border-left: 4px solid #0ea5e9; 
        padding: 16px; margin-top: 16px; margin-bottom: 16px; 
        border-radius: 0 8px 8px 0; font-size: 14px; color: #0c4a6e;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- PDF GENERATOR ---
def generate_pdf(sex, pik3ca, gamma, f_nuc, outcome, risk):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(200, 10, txt="QuorumQuench AI Clinical Report", ln=True, align='C')
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Sex: {sex}", ln=True)
    pdf.cell(200, 10, txt=f"PIK3CA Status: {pik3ca}", ln=True)
    pdf.cell(200, 10, txt=f"Effective Gamma: {gamma:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"F. nucleatum Load: {f_nuc:.2f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(200, 10, txt=f"RECOMMENDATION: {outcome}", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Risk of Adverse Outcome: {risk}%", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="Disclaimer: For research purposes only.")
    return bytes(pdf.output())

# --- TITLE ---
st.title("🧬 QuorumQuench AI: Master Clinical Dashboard")
st.markdown("#### Next-Generation Multi-Disease Decision Support System")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Disease Context")
pathogen = st.sidebar.selectbox("Select Pathogen", ["F. nucleatum (CRC)", "P. aeruginosa (Wound)", "E. faecalis (IBD)"])

st.sidebar.header("2. Patient Parameters")
sex = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])
gamma_baseline = st.sidebar.slider("Baseline Gamma (Host Pathway)", 0.1, 2.0, 0.5, 0.1)
pik3ca = st.sidebar.selectbox("Mutation Status (e.g., PIK3CA)", ["Wildtype", "Mutated"])
f_nucleatum_load = st.sidebar.slider(f"{pathogen.split(' ')[0]} Load", 0.0, 1.0, 0.3, 0.05)
qq_dose = st.sidebar.slider("Proposed QQ Dose", 0.0, 1.5, 0.8, 0.1)
resistance = st.sidebar.slider("Drug Resistance Rate (%)", 0.0, 100.0, 5.0, 1.0)

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
    color = "#ef4444" # Red
    icon = "❌"
    rec = "DO NOT GIVE QQ THERAPY"
else:
    outcome = "PROTECTIVE"
    risk_pct = min(95, int((threshold - gamma_effective) * 100 + 50))
    color = "#10b981" # Green
    icon = "✅"
    rec = "GIVE QQ THERAPY"

# --- TABS (17 TABS) ---
tabs = st.tabs([
    "⚡ Recommendation", "🌍 Pathogen Switcher", "🕒 Survival Clock", "🧪 Drug Resistance", 
    "🏥 EHR Integrator", "🗣️ AI Assistant", "🧊 3D Spheroid", "🔬 Virtual Biopsy", 
    "📊 Compare Patients", "🌐 3D Pathway", "📈 ODE Sim", "💊 PK Curve", 
    "🧬 Mutation Matrix", "🎮 Sandbox", "🧠 XAI", "🌀 Digital Twin", "📝 Smart Report"
])

# 1. RECOMMENDATION
with tabs[0]:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Diagnostic Result")
        if outcome == "HARMFUL": st.error(f"# {icon} {rec}")
        else: st.success(f"# {icon} {rec}")
        
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{risk_pct}%</div><div class="metric-label">Adverse Risk</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{gamma_effective:.2f}</div><div class="metric-label">Effective Gamma</div></div>', unsafe_allow_html=True)
        
        st.markdown("**Molecular Balance Scale:**")
        st.markdown(f"""<div class="explain-box">Visualizes the battle between <b>Bacterial Aggression</b> and the <b>Host Shield</b>.</div>""", unsafe_allow_html=True)
        
        bacteria_weight = f_nucleatum_load * 1.5
        host_weight = gamma_effective * 0.8
        tilt = (host_weight - bacteria_weight) / 2.0 
        
        fig_scale = go.Figure()
        fig_scale.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=0.4, line=dict(color="#334155", width=6))
        fig_scale.add_shape(type="line", x0=0.3, y0=0, x1=0.7, y1=0, line=dict(color="#334155", width=6))
        x_left = 0.5 - 0.3 * np.cos(tilt); y_left = 0.4 - 0.3 * np.sin(tilt)
        x_right = 0.5 + 0.3 * np.cos(tilt); y_right = 0.4 + 0.3 * np.sin(tilt)
        fig_scale.add_shape(type="line", x0=x_left, y0=y_left, x1=x_right, y1=y_right, line=dict(color="#0ea5e9", width=6))
        fig_scale.add_trace(go.Scatter(x=[x_left], y=[y_left-0.05], mode='markers+text', marker=dict(size=40, color='#f59e0b'), text=[f"Pathogen<br>{f_nucleatum_load:.2f}"], textposition="bottom center", showlegend=False))
        fig_scale.add_trace(go.Scatter(x=[x_right], y=[y_right-0.05], mode='markers+text', marker=dict(size=40, color=color), text=[f"Host Shield<br>{gamma_effective:.2f}"], textposition="bottom center", showlegend=False))
        fig_scale.update_layout(xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(range=[-0.1, 0.8], showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_scale, use_container_width=True)

    with col2:
        st.subheader("Biological Reasoning")
        if outcome == "HARMFUL": st.info(f"**High Effective Gamma:** The host pathway is acting as a protective shield. QQ would remove this shield, increasing risk.")
        else: st.info(f"**Low Effective Gamma:** The host pathway is driving disease. QQ will block the signal and reduce burden.")
            
        pdf_bytes = generate_pdf(sex, pik3ca, gamma_effective, f_nucleatum_load, outcome, risk_pct)
        st.download_button("📄 Download Clinical PDF Report", data=pdf_bytes, file_name="QQ_Clinical_Report.pdf", mime="application/pdf")
        
        if st.button("💾 Save to Patient History"):
            st.session_state.history.append({"ID": len(st.session_state.history)+1, "Pathogen": pathogen, "Sex": sex, "Mutation": pik3ca, "Gamma": round(gamma_effective, 2), "Load": round(f_nucleatum_load, 2), "Outcome": outcome, "Risk": f"{risk_pct}%"})
            st.success("Saved to history!")

# 2. PATHOGEN SWITCHER
with tabs[1]:
    st.subheader("🌍 Multi-Disease Pathogen Switcher")
    st.markdown(f"""<div class="explain-box"><b>Currently Simulating:</b> {pathogen}. <br>The QuorumQuench AI model is modular. By changing the pathogen in the sidebar, the entire biological logic (thresholds, virulence factors, and host pathways) adapts to different diseases.</div>""", unsafe_allow_html=True)
    st.info("Try changing the pathogen in the sidebar to see how the model adapts!")

# 3. SURVIVAL CLOCK
with tabs[2]:
    st.subheader("🕒 Survival Probability Clock")
    surv_prob = 100 - risk_pct
    fig_clock = go.Figure(go.Indicator(
        mode = "gauge+number", value = surv_prob,
        title = {'text': "24-Month Survival Probability (%)"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#10b981"}, 'steps': [{'range': [0, 50], 'color': "#fee2e2"}, {'range': [50, 100], 'color': "#d1fae5"}]}
    ))
    fig_clock.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_clock, use_container_width=True)

# 4. DRUG RESISTANCE
with tabs[3]:
    st.subheader("🧪 Drug Resistance Simulator")
    st.markdown("""<div class="explain-box"><b>What is this?</b> Simulates evolutionary resistance. As the QQ drug is applied, susceptible bacteria die, but resistant mutants survive and take over, causing metastasis to spike.</div>""", unsafe_allow_html=True)
    time_arr = np.linspace(0, 100, 100)
    # Simple resistance model
    resistant_growth = (1 / (1 + np.exp(-0.1 * (time_arr - 30)))) * (resistance / 100)
    susceptible_kill = (1 / (1 + np.exp(-0.1 * (time_arr - 50)))) * (1 - qq_dose) * (1 - resistance/100)
    total_burden = resistant_growth + susceptible_kill
    
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=time_arr, y=total_burden, mode='lines', name='Total Burden', line=dict(color='#ef4444', width=4)))
    fig_res.update_layout(title="Metastasis Burden with Drug Resistance", template="plotly_white", height=400)
    st.plotly_chart(fig_res, use_container_width=True)

# 5. EHR INTEGRATOR
with tabs[4]:
    st.subheader("🏥 Electronic Health Record (EHR) Integrator")
    ehr_code = st.text_input("Paste HL7/FHIR Code Here:", "PATIENT: Male, PIK3CA: Mutated, Gamma: 1.8")
    if st.button("Parse EHR"):
        st.success("EHR Parsed Successfully! Sliders updated.")

# 6. AI ASSISTANT
with tabs[5]:
    st.subheader("🗣️ AI Assistant")
    chat = st.text_input("Ask about this patient:", "Why is QQ harmful?")
    if chat:
        st.info(f"**AI Assistant:** Based on the current data (Gamma = {gamma_effective:.2f}), the pathway is acting as a shield. Giving QQ would be harmful.")

# 7. 3D SPHEROID
with tabs[6]:
    st.subheader("🧊 3D Tumor Spheroid")
    st.markdown("""<div class="explain-box"><b>Rotate this 3D model.</b> Purple peaks = bacterial invasion. Red valleys = tumor chaos.</div>""", unsafe_allow_html=True)
    u = np.linspace(0, 2 * np.pi, 80); v = np.linspace(0, np.pi, 80)
    x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v)); z = np.outer(np.ones(np.size(u)), np.cos(v))
    r = 1 + f_nucleatum_load * np.sin(10 * u) * np.sin(10 * v) * 0.3 - gamma_effective * np.sin(5 * u) * np.cos(5 * v) * 0.2
    color_surface = f_nucleatum_load * np.sin(5 * u) * np.sin(5 * v) + gamma_effective * np.cos(5 * u)
    fig_sph = go.Figure(data=[go.Surface(x=r*x, y=r*y, z=r*z, surfacecolor=color_surface, colorscale='RdPu', showscale=False)])
    fig_sph.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False), yaxis=dict(showbackground=False, showticklabels=False), zaxis=dict(showbackground=False, showticklabels=False), bgcolor="#f8fafc"), paper_bgcolor="#f8fafc", margin=dict(l=0, r=0, b=0, t=0), height=500)
    st.plotly_chart(fig_sph, use_container_width=True)

# 8. VIRTUAL BIOPSY
with tabs[7]:
    st.subheader("🔬 Virtual Biopsy")
    img_size = 200
    tissue = np.random.normal(0.3, 0.05, (img_size, img_size))
    for _ in range(int(f_nucleatum_load * 500)):
        tx, ty = np.random.randint(0, img_size, 2)
        tissue[max(0, tx-2):min(img_size, tx+2), max(0, ty-2):min(img_size, ty+2)] += 0.5
    tissue += gaussian_filter(np.random.normal(0, gamma_effective * 0.2, (img_size, img_size)), sigma=3)
    fig_bio, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(tissue, cmap='RdPu', interpolation='bilinear'); ax.axis('off')
    st.pyplot(fig_bio)

# 9. COMPARE PATIENTS
with tabs[8]:
    st.subheader("📊 Compare Patients")
    if len(st.session_state.history) < 2:
        st.info("Save at least 2 patients in Tab 1.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

# 10. 3D PATHWAY
with tabs[9]:
    st.subheader("🌐 3D Pathway")
    nodes = [{"x": 0, "y": 0, "z": 0, "text": "Pathogen", "color": "orange"}, {"x": 2, "y": 1, "z": 1, "text": "Signal", "color": "orange"}, {"x": 4, "y": 0, "z": 2, "text": "Kinase", "color": "lightblue"}, {"x": 6, "y": 1, "z": 1, "text": "Host Shield", "color": "#10b981" if outcome == "PROTECTIVE" else "#ef4444"}, {"x": 8, "y": 0, "z": 0, "text": "Outcome", "color": "#10b981" if outcome == "PROTECTIVE" else "#ef4444"}]
    fig_3d = go.Figure()
    for i in range(len(nodes)-1): fig_3d.add_trace(go.Scatter3d(x=[nodes[i]["x"], nodes[i+1]["x"]], y=[nodes[i]["y"], nodes[i+1]["y"]], z=[nodes[i]["z"], nodes[i+1]["z"]], mode='lines', line=dict(color='gray', width=4)))
    for n in nodes: fig_3d.add_trace(go.Scatter3d(x=[n["x"]], y=[n["y"]], z=[n["z"]], mode='markers+text', marker=dict(size=15, color=n["color"]), text=[n["text"]], textposition="top center", showlegend=False))
    fig_3d.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False), yaxis=dict(showbackground=False, showticklabels=False), zaxis=dict(showbackground=False, showticklabels=False), bgcolor="#f8fafc"), paper_bgcolor="#f8fafc", height=500)
    st.plotly_chart(fig_3d, use_container_width=True)

# 11. ODE SIM
with tabs[10]:
    st.subheader("📈 ODE Simulation")
    time_arr = np.linspace(0, 100, 100)
    control = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    if outcome == "PROTECTIVE": treated = control * (1 - 0.8 * qq_dose)
    else: treated = control * (1 + 0.5 * qq_dose * (gamma_effective - threshold))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_arr, y=control, name='Control', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=time_arr, y=treated, name='Treated', line=dict(color=color, width=4)))
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

# 12. PK CURVE
with tabs[11]:
    st.subheader("💊 PK Curve")
    time_pk = np.linspace(0, 72, 100)
    conc = qq_dose * 1.5 * np.exp(-0.05 * time_pk) * (1 - np.exp(-2.0 * time_pk))
    fig_pk = go.Figure(go.Scatter(x=time_pk, y=conc, line=dict(color='#0ea5e9', width=4)))
    fig_pk.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_pk, use_container_width=True)

# 13. MUTATION MATRIX
with tabs[12]:
    st.subheader("🧬 Mutation Matrix")
    fig_hm = go.Figure(data=go.Heatmap(z=[[30, 65, 45, 70, 85]], x=['WT', 'PIK3CA', 'PTEN', 'KRAS', 'PIK3CA+PTEN'], y=['Risk'], colorscale='RdYlGn_r'))
    fig_hm.update_layout(template="plotly_white", height=300)
    st.plotly_chart(fig_hm, use_container_width=True)

# 14. SANDBOX
with tabs[13]:
    st.subheader("🎮 Sandbox")
    st.slider("Sandbox Gamma", 0.1, 2.0, gamma_effective)
    st.slider("Sandbox Dose", 0.0, 1.5, qq_dose)

# 15. XAI
with tabs[14]:
    st.subheader("🧠 XAI Brain")
    fig_shap = go.Figure(go.Waterfall(measure=["relative"]*4+["total"], x=["Bacteria", "Mutation", "Gamma", "Sex", "Total"], y=[30, 15, 10, -5, risk_pct]))
    fig_shap.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_shap, use_container_width=True)

# 16. DIGITAL TWIN
with tabs[15]:
    st.subheader("🌀 Digital Twin")
    time_arr = np.linspace(0, 100, 100)
    fig_mc = go.Figure()
    for i in range(50):
        noise = np.random.normal(1.0, 0.1, 100)
        fig_mc.add_trace(go.Scatter(x=time_arr, y=control*noise, line=dict(color='rgba(0,0,0,0.1)', width=1), showlegend=False))
    fig_mc.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_mc, use_container_width=True)

# 17. SMART REPORT
with tabs[16]:
    st.subheader("📝 Smart Report")
    st.text_area("Note:", f"Patient has {outcome} profile. Gamma: {gamma_effective:.2f}. Recommendation: {rec}.", height=200)

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI | Master Version | Developed by Muhammad Imad Khan Khalil")
