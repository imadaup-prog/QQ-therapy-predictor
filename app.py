import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.draw import disk, ellipse
from skimage.util import random_noise
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QuorumQuench AI", page_icon="🧬", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; }
    .metric-card { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; margin-bottom: 20px; border: 1px solid #e2e8f0; }
    .metric-value { font-size: 2.5rem; font-weight: 800; color: #0f172a; }
    .metric-label { font-size: 0.9rem; color: #64748b; text-transform: uppercase; }
    .stButton>button { width: 100%; height: 3em; font-size: 16px; font-weight: 600; background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%); color: white; border-radius: 8px; border: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; overflow-x: auto; flex-wrap: nowrap; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #f1f5f9; border-radius: 6px 6px 0 0; padding: 10px 16px; font-weight: 500; color: #475569; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; font-weight: 700; color: #0ea5e9; border-bottom: 3px solid #0ea5e9; }
    .explain-box { background-color: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 16px; margin-top: 16px; margin-bottom: 16px; border-radius: 0 8px 8px 0; font-size: 14px; color: #0c4a6e; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- TITLE ---
st.title("🧬 QuorumQuench AI")
st.markdown("#### Precision Oncology Decision Support System")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Patient Parameters")
sex = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])
gamma_baseline = st.sidebar.slider("Baseline Gamma (p-AKT Activity)", 0.1, 2.0, 0.5, 0.1)
pik3ca = st.sidebar.selectbox("PIK3CA Mutation Status", ["Wildtype", "Mutated"])
f_nucleatum_load = st.sidebar.slider("F. nucleatum Abundance", 0.0, 1.0, 0.3, 0.05)
qq_dose = st.sidebar.slider("Proposed QQ Dose", 0.0, 1.5, 0.8, 0.1)

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
    color = "#ef4444"
    icon = "❌"
    rec = "DO NOT GIVE QQ THERAPY"
else:
    outcome = "PROTECTIVE"
    risk_pct = min(95, int((threshold - gamma_effective) * 100 + 50))
    color = "#10b981"
    icon = "✅"
    rec = "GIVE QQ THERAPY"

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "⚡ Recommendation", "🎮 Sandbox", "🧠 XAI", "🧊 3D Spheroid", 
    "🌐 3D Pathway (Fixed)", "🔬 Virtual Biopsy (Fixed)", "📊 Compare", "📝 Report"
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
        st.markdown("""<div class="explain-box"><b>How to read this:</b> The scale shows the battle between Bacteria (left) and Host Shield (right). If the Host Shield is heavier, the pathway is protective and QQ should be avoided.</div>""", unsafe_allow_html=True)
        
        bacteria_weight = f_nucleatum_load * 1.5; host_weight = gamma_effective * 0.8
        tilt = (host_weight - bacteria_weight) / 2.0 
        fig_scale = go.Figure()
        fig_scale.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=0.4, line=dict(color="#334155", width=6))
        fig_scale.add_shape(type="line", x0=0.3, y0=0, x1=0.7, y1=0, line=dict(color="#334155", width=6))
        x_left = 0.5 - 0.3 * np.cos(tilt); y_left = 0.4 - 0.3 * np.sin(tilt)
        x_right = 0.5 + 0.3 * np.cos(tilt); y_right = 0.4 + 0.3 * np.sin(tilt)
        fig_scale.add_shape(type="line", x0=x_left, y0=y_left, x1=x_right, y1=y_right, line=dict(color="#0ea5e9", width=6))
        fig_scale.add_trace(go.Scatter(x=[x_left], y=[y_left-0.05], mode='markers+text', marker=dict(size=40, color='#f59e0b'), text=[f"F. nucleatum<br>{f_nucleatum_load:.2f}"], textposition="bottom center", showlegend=False))
        fig_scale.add_trace(go.Scatter(x=[x_right], y=[y_right-0.05], mode='markers+text', marker=dict(size=40, color=color), text=[f"Gamma<br>{gamma_effective:.2f}"], textposition="bottom center", showlegend=False))
        fig_scale.update_layout(xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(range=[-0.1, 0.8], showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_scale, use_container_width=True)

    with col2:
        st.subheader("Biological Reasoning")
        if outcome == "HARMFUL": st.info("**High Effective Gamma:** The PI3K/AKT pathway is acting as a protective shield. QQ would remove this shield, increasing metastasis risk.")
        else: st.info("**Low Effective Gamma:** The PI3K/AKT pathway is driving metastasis. QQ will block the AI-2 signal and reduce tumor burden.")
        if st.button("💾 Save to Patient History"):
            st.session_state.history.append({"ID": len(st.session_state.history)+1, "Sex": sex, "PIK3CA": pik3ca, "Gamma": round(gamma_effective, 2), "F. nucleatum": round(f_nucleatum_load, 2), "Outcome": outcome})
            st.success("Saved!")

# ==========================================
# TAB 2: SANDBOX
# ==========================================
with tab2:
    st.subheader("🎮 Treatment Sandbox")
    st.markdown("""<div class="explain-box"><b>How to use:</b> Adjust the sliders to simulate hypothetical patients. If the colored line goes UP, QQ is harmful. If it goes DOWN, QQ is protective.</div>""", unsafe_allow_html=True)
    sandbox_qq = st.slider("Adjust QQ Dose", 0.0, 1.5, qq_dose, 0.1)
    sandbox_gamma = st.slider("Adjust Gamma Level", 0.1, 2.0, gamma_effective, 0.1)
    time_arr = np.linspace(0, 100, 100)
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    if sandbox_gamma > threshold:
        sandbox_metastasis = control_metastasis * (1 + 0.5 * sandbox_qq * (sandbox_gamma - threshold)); sandbox_color = "#ef4444"
    else:
        sandbox_metastasis = control_metastasis * (1 - 0.8 * sandbox_qq); sandbox_color = "#10b981"
    fig_sandbox = go.Figure()
    fig_sandbox.add_trace(go.Scatter(x=time_arr, y=control_metastasis, name='No QQ', line=dict(color='gray', dash='dash')))
    fig_sandbox.add_trace(go.Scatter(x=time_arr, y=sandbox_metastasis, name='QQ Treated', line=dict(color=sandbox_color, width=4)))
    fig_sandbox.update_layout(title="Real-Time Metastasis Trajectory", template="plotly_white", height=400)
    st.plotly_chart(fig_sandbox, use_container_width=True)
    if sandbox_gamma > threshold: st.error(f"⚠️ At Gamma = {sandbox_gamma:.2f}, QQ is HARMFUL.")
    else: st.success(f"✅ At Gamma = {sandbox_gamma:.2f}, QQ is PROTECTIVE.")

# ==========================================
# TAB 3: XAI
# ==========================================
with tab3:
    st.subheader("🧠 AI Explainability (XAI)")
    st.markdown("""<div class="explain-box"><b>What is this?</b> This Waterfall plot explains the "Brain" of the AI. It shows how each factor pushes risk UP (red) or DOWN (green).</div>""", unsafe_allow_html=True)
    shap_values = {"F. nucleatum Load": f_nucleatum_load * 30, "PIK3CA Mutation": 15 if pik3ca == "Mutated" else 0, "Baseline Gamma": gamma_baseline * 10, "Sex (S0)": 10 if sex == "Female" else -5, "QQ Dose Effect": -20 if outcome == "PROTECTIVE" else 20}
    fig_shap = go.Figure(go.Waterfall(
        name="SHAP", orientation="v", measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=list(shap_values.keys()) + ["Total Risk"], textposition="outside", text=[f"{v:+.1f}%" for v in shap_values.values()] + [f"{risk_pct}%"],
        y=list(shap_values.values()) + [risk_pct], connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"#10b981"}}, increasing={"marker":{"color":"#ef4444"}}, totals={"marker":{"color":"#0ea5e9"}}
    ))
    fig_shap.update_layout(title="Feature Contribution to Risk", template="plotly_white", waterfallgap=0.3, height=500)
    st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# TAB 4: 3D SPHEROID
# ==========================================
with tab4:
    st.subheader("🧊 3D Tumor Spheroid")
    st.markdown("""<div class="explain-box"><b>Rotate this 3D model.</b> Purple peaks = bacterial invasion. Red valleys = tumor chaos.</div>""", unsafe_allow_html=True)
    u = np.linspace(0, 2 * np.pi, 100); v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v)); z = np.outer(np.ones(np.size(u)), np.cos(v))
    r = 1 + f_nucleatum_load * np.sin(10 * u) * np.sin(10 * v) * 0.3 - gamma_effective * np.sin(5 * u) * np.cos(5 * v) * 0.2
    color_surface = f_nucleatum_load * np.sin(5 * u) * np.sin(5 * v) + gamma_effective * np.cos(5 * u)
    fig_sph = go.Figure(data=[go.Surface(x=r*x, y=r*y, z=r*z, surfacecolor=color_surface, colorscale='RdPu', showscale=False)])
    fig_sph.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False), yaxis=dict(showbackground=False, showticklabels=False), zaxis=dict(showbackground=False, showticklabels=False), bgcolor="#f8fafc"), paper_bgcolor="#f8fafc", margin=dict(l=0, r=0, b=0, t=0), height=500)
    st.plotly_chart(fig_sph, use_container_width=True)

# ==========================================
# TAB 5: 3D PATHWAY (FIXED WITH COLORS)
# ==========================================
with tab5:
    st.subheader("🌐 3D Interactive Biological Pathway")
    st.markdown("""<div class="explain-box"><b>Step-by-Step Activation:</b><br>1. F. nucleatum (orange) adheres to the colon.<br>2. Virulence Factors (orange) are injected.<br>3. PI3K (blue) is activated.<br>4. AKT (Gamma) (green/red) creates the protective shield.<br>5. Metastasis is the final outcome.<br><i>Rotate the 3D graph to see the spatial relationship.</i></div>""", unsafe_allow_html=True)
    
    # Node coordinates and colors
    nodes = [
        {"x": 0, "y": 0, "z": 0, "text": f"F. nucleatum<br>Load: {f_nucleatum_load:.2f}", "color": "#f59e0b", "size": 20},
        {"x": 2, "y": 1, "z": 1, "text": "Fap2 / RadD<br>(Virulence)", "color": "#f97316", "size": 15},
        {"x": 4, "y": 0, "z": 2, "text": "PI3K<br>(Activation)", "color": "#3b82f6", "size": 15},
        {"x": 6, "y": 1, "z": 1, "text": f"AKT (Gamma)<br>Score: {gamma_effective:.2f}", "color": "#10b981" if outcome == "PROTECTIVE" else "#ef4444", "size": 25},
        {"x": 8, "y": 0, "z": 0, "text": f"Metastasis<br>{rec}", "color": "#10b981" if outcome == "PROTECTIVE" else "#ef4444", "size": 20}
    ]
    
    fig_3d = go.Figure()
    
    # Draw arrows (cones) between nodes
    for i in range(len(nodes)-1):
        fig_3d.add_trace(go.Cone(
            x=[nodes[i+1]["x"]], y=[nodes[i+1]["y"]], z=[nodes[i+1]["z"]],
            u=[nodes[i+1]["x"] - nodes[i]["x"]], v=[nodes[i+1]["y"] - nodes[i]["y"]], w=[nodes[i+1]["z"] - nodes[i]["z"]],
            sizemode="absolute", sizeref=0.5, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']]
        ))
    
    # Draw nodes
    for n in nodes:
        fig_3d.add_trace(go.Scatter3d(
            x=[n["x"]], y=[n["y"]], z=[n["z"]],
            mode='markers+text',
            marker=dict(size=n["size"], color=n["color"], opacity=0.9, line=dict(color='white', width=2)),
            text=[n["text"]],
            textposition="top center",
            textfont=dict(size=12, color="black"),
            showlegend=False
        ))
    
    # Add background gradient and grid
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(showbackground=True, backgroundcolor="#e0f2fe", showgrid=True, gridcolor="white", title=''),
            yaxis=dict(showbackground=True, backgroundcolor="#e0f2fe", showgrid=True, gridcolor="white", title=''),
            zaxis=dict(showbackground=True, backgroundcolor="#e0f2fe", showgrid=True, gridcolor="white", title=''),
            bgcolor="#f8fafc"
        ),
        paper_bgcolor="#f8fafc",
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# TAB 6: VIRTUAL BIOPSY (FIXED - REALISTIC TISSUE)
# ==========================================
with tab6:
    st.subheader("🔬 Virtual Biopsy: H&E Histology Simulation")
    st.markdown("""<div class="explain-box"><b>What is this?</b> A simulated H&E stain of colon tissue. <b>Purple dots</b> = F. nucleatum bacteria. <b>Pink circles</b> = colon crypts (glands). As Gamma increases, the crypts become irregular and chaotic (tumor).</div>""", unsafe_allow_html=True)
    
    # Create a realistic tissue image
    img_size = 400
    image = np.zeros((img_size, img_size, 3))
    
    # Pink background (eosin stain)
    image[:, :, 0] = 0.95 # Red
    image[:, :, 1] = 0.75 # Green
    image[:, :, 2] = 0.85 # Blue
    
    # Draw colon crypts (glands)
    num_crypts = 20
    for _ in range(num_crypts):
        r = np.random.randint(15, 40)
        c = np.random.randint(r, img_size - r)
        rr, cc = disk((c, c), r, shape=image.shape[:2])
        # Crypt lumen (white)
        image[rr, cc, 0] = 0.98; image[rr, cc, 1] = 0.95; image[rr, cc, 2] = 0.98
        # Crypt wall (purple)
        rr_outer, cc_outer = disk((c, c), r + 5, shape=image.shape[:2])
        wall_mask = np.zeros_like(image[:, :, 0], dtype=bool)
        wall_mask[rr_outer, cc_outer] = True
        wall_mask[rr, cc] = False
        image[wall_mask, 0] = 0.4; image[wall_mask, 1] = 0.2; image[wall_mask, 2] = 0.5
    
    # Add bacteria (purple dots)
    num_bacteria = int(f_nucleatum_load * 500)
    for _ in range(num_bacteria):
        r = np.random.randint(1, 4)
        c = np.random.randint(r, img_size - r)
        rr, cc = disk((c, c), r, shape=image.shape[:2])
        image[rr, cc, 0] = 0.3; image[rr, cc, 1] = 0.1; image[rr, cc, 2] = 0.4
    
    # Add tumor chaos (irregular texture) based on Gamma
    if gamma_effective > threshold:
        noise = np.random.normal(0, gamma_effective * 0.1, (img_size, img_size))
        for ch in range(3):
            image[:, :, ch] += gaussian_filter(noise, sigma=2)
    
    # Clip and show
    image = np.clip(image, 0, 1)
    fig_bio, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.axis('off')
    ax.set_title("Simulated H&E Stain (40x Magnification)", fontsize=14)
    st.pyplot(fig_bio)

# ==========================================
# TAB 7: COMPARE
# ==========================================
with tab7:
    st.subheader("📊 Compare Patients")
    st.markdown("""<div class="explain-box"><b>How to use:</b> Save at least 2 patients in Tab 1. This scatter plot shows how they cluster.</div>""", unsafe_allow_html=True)
    if len(st.session_state.history) < 2: st.info("Save at least 2 patients in Tab 1.")
    else:
        df = pd.DataFrame(st.session_state.history)
        fig_comp = go.Figure()
        for i, row in df.iterrows():
            fig_comp.add_trace(go.Scatter(x=[row['Gamma']], y=[row['F. nucleatum']], mode='markers+text', marker=dict(size=25, color='#ef4444' if row['Outcome'] == 'HARMFUL' else '#10b981', line=dict(color='black', width=2)), text=[f"Patient {row['ID']}"], textposition="top center"))
        fig_comp.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_comp.update_layout(title="Patient Clustering", template="plotly_white", height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(df, use_container_width=True)

# ==========================================
# TAB 8: SMART REPORT
# ==========================================
with tab8:
    st.subheader("📝 Smart Report")
    st.markdown("""<div class="explain-box"><b>Auto-generated clinical note.</b></div>""", unsafe_allow_html=True)
    report = f"Patient: {sex}, PIK3CA {pik3ca}. Effective Gamma: {gamma_effective:.2f}. F. nucleatum Load: {f_nucleatum_load:.2f}.\n\nAI RECOMMENDATION: {outcome} (Risk: {risk_pct}%)."
    st.text_area("Generated Clinical Note:", report, height=200)
    st.download_button("📄 Download Note", data=report, file_name="Clinical_Note.txt")

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI | Master Version | Developed by Muhammad Imad Khan Khalil")
