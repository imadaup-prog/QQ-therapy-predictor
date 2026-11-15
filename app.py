import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# --- TABS (6 TABS) ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ Recommendation", "🎮 Sandbox", "🧠 XAI", 
    "🫁 3D Human Colon", "📊 Compare", "📝 Report"
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
            st.session_state.history.append({"ID": len(st.session_state.history)+1, "Sex": sex, "PIK3CA": pik3ca, "Gamma": round(gamma_effective, 2), "F. nucleatum": round(f_nucleatum_load, 2), "Outcome": outcome, "Risk": f"{risk_pct}%"})
            st.success("Saved!")

# ==========================================
# TAB 2: SANDBOX (Two Graphs Restored)
# ==========================================
with tab2:
    st.subheader("🎮 Treatment Sandbox: Real-Time What-If Analysis")
    st.markdown("""
    <div class="explain-box">
    <b>How to use:</b> Adjust the sliders to simulate hypothetical patients. 
    <br><b>Top Graph:</b> Shows how the QQ drug suppresses the bacterial load.
    <br><b>Bottom Graph:</b> Shows the downstream effect on metastasis. 
    If the colored line goes UP, QQ is harmful. If it goes DOWN, QQ is protective.
    </div>
    """, unsafe_allow_html=True)
    
    sandbox_qq = st.slider("Adjust QQ Dose", 0.0, 1.5, qq_dose, 0.1)
    sandbox_gamma = st.slider("Adjust Gamma Level", 0.1, 2.0, gamma_effective, 0.1)
    
    time_arr = np.linspace(0, 100, 100)
    
    # --- Graph 1: Bacterial Load ---
    b_growth = 1 / (1 + np.exp(-0.1 * (time_arr - 30))) * f_nucleatum_load
    b_qq = b_growth * (1 - 0.9 * sandbox_qq)
    
    fig_bacteria = go.Figure()
    fig_bacteria.add_trace(go.Scatter(x=time_arr, y=b_growth, mode='lines', name='F. nucleatum (No QQ)', line=dict(color='orange', dash='dash', width=2)))
    fig_bacteria.add_trace(go.Scatter(x=time_arr, y=b_qq, mode='lines', name='F. nucleatum (QQ Treated)', line=dict(color='purple', width=4)))
    fig_bacteria.update_layout(
        title="Bacterial Load Dynamics",
        xaxis_title="Time (Days)",
        yaxis_title="Relative Abundance",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig_bacteria, use_container_width=True)
    
    # --- Graph 2: Metastasis ---
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    if sandbox_gamma > threshold:
        sandbox_metastasis = control_metastasis * (1 + 0.5 * sandbox_qq * (sandbox_gamma - threshold))
        sandbox_color = "#ef4444"
    else:
        sandbox_metastasis = control_metastasis * (1 - 0.8 * sandbox_qq)
        sandbox_color = "#10b981"
    
    fig_metastasis = go.Figure()
    fig_metastasis.add_trace(go.Scatter(x=time_arr, y=control_metastasis, mode='lines', name='Metastasis (No QQ)', line=dict(color='gray', dash='dash', width=2)))
    fig_metastasis.add_trace(go.Scatter(x=time_arr, y=sandbox_metastasis, mode='lines', name='Metastasis (QQ Treated)', line=dict(color=sandbox_color, width=4)))
    fig_metastasis.update_layout(
        title="Metastasis Burden",
        xaxis_title="Time (Days)",
        yaxis_title="Metastasis Burden",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig_metastasis, use_container_width=True)
    
    # --- Result ---
    if sandbox_gamma > threshold:
        st.error(f"⚠️ **Result:** At Gamma = {sandbox_gamma:.2f} (above threshold of {threshold:.2f}), QQ therapy is **HARMFUL** because it removes the protective pathway.")
    else:
        st.success(f"✅ **Result:** At Gamma = {sandbox_gamma:.2f} (below threshold of {threshold:.2f}), QQ therapy is **PROTECTIVE** because it blocks the pro-metastatic signal.")

# ==========================================
# TAB 3: XAI
# ==========================================
with tab3:
    st.subheader("🧠 AI Explainability (XAI)")
    st.markdown("""<div class="explain-box"><b>What is this?</b> This Waterfall plot explains the "Brain" of the AI. It shows how each factor pushes risk UP (red) or DOWN (green).</div>""", unsafe_allow_html=True)
    mutation_label = f"PIK3CA ({pik3ca})"
    mutation_value = 15 if pik3ca == "Mutated" else 0
    shap_values = {"F. nucleatum Load": f_nucleatum_load * 30, mutation_label: mutation_value, "Baseline Gamma": gamma_baseline * 10, f"Sex ({sex})": 10 if sex == "Female" else -5, "QQ Dose Effect": -20 if outcome == "PROTECTIVE" else 20}
    fig_shap = go.Figure(go.Waterfall(
        name="SHAP", orientation="v", measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=list(shap_values.keys()) + ["Total Risk"], textposition="outside", text=[f"{v:+.1f}%" for v in shap_values.values()] + [f"{risk_pct}%"],
        y=list(shap_values.values()) + [risk_pct], connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"#10b981"}}, increasing={"marker":{"color":"#ef4444"}}, totals={"marker":{"color":"#0ea5e9"}}
    ))
    fig_shap.update_layout(title="Feature Contribution to Risk", template="plotly_white", waterfallgap=0.3, height=500)
    st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# TAB 4: 3D HUMAN COLON (Interactive)
# ==========================================
with tab4:
    st.subheader("🫁 3D Human Colon: Molecular Pathway Visualization")
    st.markdown("""<div class="explain-box"><b>What is this?</b> A 3D holographic view of the human colon. Inside the tissue, you can see the biological pathway: <b>F. nucleatum</b> (orange), <b>Virulence Factors</b> (yellow), <b>PI3K</b> (blue), <b>AKT/Gamma</b> (green/red), and <b>Metastasis</b> (green/red). Rotate the graph with your mouse to explore.</div>""", unsafe_allow_html=True)
    
    # Create the 3D scene
    fig_colon = go.Figure()
    
    # 1. Draw the colon (a tube shape)
    t = np.linspace(0, 4*np.pi, 100)
    x_tube = 10 * np.cos(t)
    y_tube = 10 * np.sin(t)
    z_tube = np.linspace(-5, 5, 100)
    
    # Colon wall
    fig_colon.add_trace(go.Mesh3d(
        x=x_tube, y=y_tube, z=z_tube,
        alphahull=0, opacity=0.15, color='#fda4af', name='Colon Tissue'
    ))
    
    # 2. Add the pathway nodes inside the colon
    nodes = [
        {"x": 10, "y": 0, "z": -3, "text": f"F. nucleatum<br>Load: {f_nucleatum_load:.2f}", "color": "#f59e0b", "size": 15},
        {"x": 0, "y": 10, "z": -1.5, "text": "Fap2 / RadD<br>(Virulence)", "color": "#f97316", "size": 12},
        {"x": -10, "y": 0, "z": 0, "text": "PI3K<br>(Activation)", "color": "#3b82f6", "size": 12},
        {"x": 0, "y": -10, "z": 1.5, "text": f"AKT (Gamma)<br>{gamma_effective:.2f}", "color": "#10b981" if outcome == "PROTECTIVE" else "#ef4444", "size": 18},
        {"x": 10, "y": 0, "z": 3, "text": f"Metastasis<br>{rec}", "color": "#10b981" if outcome == "PROTECTIVE" else "#ef4444", "size": 15}
    ]
    
    # Draw arrows between nodes
    for i in range(len(nodes)-1):
        fig_colon.add_trace(go.Scatter3d(
            x=[nodes[i]["x"], nodes[i+1]["x"]],
            y=[nodes[i]["y"], nodes[i+1]["y"]],
            z=[nodes[i]["z"], nodes[i+1]["z"]],
            mode='lines', line=dict(color='#64748b', width=5), showlegend=False
        ))
    
    # Draw nodes
    for n in nodes:
        fig_colon.add_trace(go.Scatter3d(
            x=[n["x"]], y=[n["y"]], z=[n["z"]],
            mode='markers+text',
            marker=dict(size=n["size"], color=n["color"], opacity=0.9, line=dict(color='white', width=2)),
            text=[n["text"]], textposition="top center",
            textfont=dict(size=11, color="black"), showlegend=False
        ))
    
    # Styling the 3D scene
    fig_colon.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor="#f8fafc",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        paper_bgcolor="#f8fafc",
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig_colon, use_container_width=True)
    
    st.info("💡 **Tip:** Use your mouse to **rotate, zoom, and pan** this 3D model. The pathway is floating inside the colon tissue.")

# ==========================================
# TAB 5: COMPARE PATIENTS
# ==========================================
with tab5:
    st.subheader("📊 Clinical Trial Mode: Side-by-Side Comparison")
    st.markdown("""<div class="explain-box"><b>How to use:</b> Save at least 2 patients in Tab 1. This tab will show a scatter plot, a comparison table, and summary statistics.</div>""", unsafe_allow_html=True)
    
    if len(st.session_state.history) < 2:
        st.info("Please save at least 2 patients to the history (Tab 1) to compare them.")
    else:
        df = pd.DataFrame(st.session_state.history)
        
        st.markdown("### Patient Clustering (Gamma vs. F. nucleatum Load)")
        fig_comp = go.Figure()
        for i, row in df.iterrows():
            fig_comp.add_trace(go.Scatter(x=[row['Gamma']], y=[row['F. nucleatum']], mode='markers+text', marker=dict(size=25, color='#ef4444' if row['Outcome'] == 'HARMFUL' else '#10b981', line=dict(color='black', width=2)), text=[f"Patient {row['ID']}"], textposition="top center"))
        fig_comp.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold:.2f})")
        fig_comp.update_layout(title="Patient Clustering", template="plotly_white", height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("### Side-by-Side Comparison Table")
        comparison_df = df[['ID', 'Sex', 'PIK3CA', 'Gamma', 'F. nucleatum', 'Outcome', 'Risk']].copy()
        comparison_df.columns = ['Patient ID', 'Sex', 'PIK3CA Status', 'Gamma', 'F. nucleatum Load', 'QQ Outcome', 'Adverse Risk']
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Cohort Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients Saved", len(df))
        c2.metric("Harmful Outcomes (❌)", len(df[df['Outcome'] == 'HARMFUL']))
        c3.metric("Protective Outcomes (✅)", len(df[df['Outcome'] == 'PROTECTIVE']))
        avg_gamma = df['Gamma'].mean()
        c4.metric("Average Gamma", f"{avg_gamma:.2f}")
        
        st.markdown("### Risk Comparison Across Patients")
        fig_bar = go.Figure(data=[
            go.Bar(x=[f"Patient {row['ID']}" for _, row in df.iterrows()], y=[int(row['Risk'].replace('%', '')) for _, row in df.iterrows()], marker_color=['#ef4444' if row['Outcome'] == 'HARMFUL' else '#10b981' for _, row in df.iterrows()], text=[row['Risk'] for _, row in df.iterrows()], textposition='auto')
        ])
        fig_bar.update_layout(title="Adverse Risk per Patient", xaxis_title="Patient", yaxis_title="Risk (%)", template="plotly_white", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 6: SMART REPORT
# ==========================================
with tab6:
    st.subheader("📝 Smart Report")
    st.markdown("""<div class="explain-box"><b>Auto-generated clinical note.</b></div>""", unsafe_allow_html=True)
    report = f"Patient: {sex}, PIK3CA {pik3ca}. Effective Gamma: {gamma_effective:.2f}. F. nucleatum Load: {f_nucleatum_load:.2f}.\n\nAI RECOMMENDATION: {outcome} (Risk: {risk_pct}%)."
    st.text_area("Generated Clinical Note:", report, height=200)
    st.download_button("📄 Download Note", data=report, file_name="Clinical_Note.txt")

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI | Master Version | Developed by Muhammad Imad Khan Khalil")
