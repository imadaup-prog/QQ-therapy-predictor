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

# ==========================================
# TRANSLATION DICTIONARY
# ==========================================
translations = {
    "English": {
        "lang_select": "Select Language",
        "title": "🧬 QuorumQuench AI",
        "subtitle": "Precision Oncology Decision Support System",
        "sidebar_header": "Patient Parameters",
        "sex": "Biological Sex", "male": "Male", "female": "Female",
        "gamma": "Baseline Gamma (p-AKT Activity)",
        "mutation": "PIK3CA Mutation Status", "wildtype": "Wildtype", "mutated": "Mutated",
        "load": "F. nucleatum Abundance", "dose": "Proposed QQ Dose",
        "tab1": "⚡ Recommendation", "tab2": "🎮 Sandbox", "tab3": "🧠 XAI",
        "tab4": "🫁 3D Human Colon", "tab5": "📊 Compare", "tab6": "📝 Report",
        "diag_res": "Diagnostic Result", "give_qq": "GIVE QQ THERAPY", "no_qq": "DO NOT GIVE QQ THERAPY",
        "adv_risk": "Adverse Risk", "eff_gamma": "Effective Gamma",
        "balance_title": "Molecular Balance Scale:",
        "balance_exp": "<b>How to read this:</b> The scale shows the battle between Bacteria (left) and Host Shield (right). If the Host Shield is heavier, the pathway is protective and QQ should be avoided.",
        "reasoning": "Biological Reasoning",
        "reason_harm": "**High Effective Gamma:** The PI3K/AKT pathway is acting as a protective shield. QQ would remove this shield, increasing metastasis risk.",
        "reason_prot": "**Low Effective Gamma:** The PI3K/AKT pathway is driving metastasis. QQ will block the AI-2 signal and reduce tumor burden.",
        "save_btn": "💾 Save to Patient History", "saved": "Saved!",
        "sandbox_title": "🎮 Treatment Sandbox: Real-Time What-If Analysis",
        "sandbox_exp": "<b>How to use:</b> Adjust the sliders to simulate hypothetical patients. <br><b>Top Graph:</b> Shows how the QQ drug suppresses the bacterial load.<br><b>Bottom Graph:</b> Shows the downstream effect on metastasis. If the colored line goes UP, QQ is harmful. If it goes DOWN, QQ is protective.",
        "sb_qq": "Adjust QQ Dose", "sb_gamma": "Adjust Gamma Level",
        "graph_bac_title": "Bacterial Load Dynamics", "graph_met_title": "Metastasis Burden",
        "x_axis_time": "Time (Days)", "y_axis_abund": "Relative Abundance", "y_axis_met": "Metastasis Burden",
        "legend_bac_no": "F. nucleatum (No QQ)", "legend_bac_yes": "F. nucleatum (QQ Treated)",
        "legend_met_no": "Metastasis (No QQ)", "legend_met_yes": "Metastasis (QQ Treated)",
        "sb_result_harm": "⚠️ **Result:** At Gamma = {:.2f} (above threshold of {:.2f}), QQ therapy is **HARMFUL** because it removes the protective pathway.",
        "sb_result_prot": "✅ **Result:** At Gamma = {:.2f} (below threshold of {:.2f}), QQ therapy is **PROTECTIVE** because it blocks the pro-metastatic signal.",
        "xai_title": "🧠 AI Explainability (XAI)",
        "xai_exp": "<b>What is this?</b> This Waterfall plot explains the Brain of the AI. It shows how each factor pushes risk UP (red) or DOWN (green).",
        "xai_xaxis": "Feature Contribution to Risk",
        "colon_title": "🫁 3D Human Colon: Molecular Pathway Visualization",
        "colon_exp": "<b>What is this?</b> A 3D holographic view of the human colon. Inside the tissue, you can see the biological pathway: <b>F. nucleatum</b> (orange), <b>Virulence Factors</b> (yellow), <b>PI3K</b> (blue), <b>AKT/Gamma</b> (green/red), and <b>Metastasis</b> (green/red). Rotate the graph with your mouse to explore.",
        "colon_tip": "💡 **Tip:** Use your mouse to **rotate, zoom, and pan** this 3D model. The pathway is floating inside the colon tissue.",
        "compare_title": "📊 Clinical Trial Mode: Side-by-Side Comparison",
        "compare_exp": "<b>How to use:</b> Save at least 2 patients in Tab 1. This tab will show a scatter plot, a comparison table, and summary statistics.",
        "compare_warn": "Please save at least 2 patients to the history (Tab 1) to compare them.",
        "compare_scatter": "### Patient Clustering (Gamma vs. F. nucleatum Load)",
        "compare_table": "### Side-by-Side Comparison Table",
        "compare_stats": "### Cohort Summary Statistics",
        "compare_bar": "### Risk Comparison Across Patients",
        "report_title": "📝 Smart Report",
        "report_exp": "<b>Auto-generated clinical note.</b>",
        "report_area": "Generated Clinical Note:",
        "report_dl": "📄 Download Note"
    },
    "中文 (Chinese)": {
        "lang_select": "选择语言",
        "title": "🧬 群体感应淬灭AI",
        "subtitle": "精准肿瘤学决策支持系统",
        "sidebar_header": "患者参数",
        "sex": "生理性别", "male": "男性", "female": "女性",
        "gamma": "基线Gamma (p-AKT活性)",
        "mutation": "PIK3CA突变状态", "wildtype": "野生型", "mutated": "突变型",
        "load": "具核梭杆菌丰度", "dose": "建议QQ剂量",
        "tab1": "⚡ 诊断建议", "tab2": "🎮 模拟沙盒", "tab3": "🧠 可解释性 (XAI)",
        "tab4": "🫁 3D人体结肠", "tab5": "📊 患者对比", "tab6": "📝 智能报告",
        "diag_res": "诊断结果", "give_qq": "建议给予QQ治疗", "no_qq": "不建议给予QQ治疗",
        "adv_risk": "不良风险", "eff_gamma": "有效Gamma",
        "balance_title": "分子平衡天平：",
        "balance_exp": "<b>如何解读：</b> 天平展示了细菌（左）与宿主保护盾（右）之间的较量。如果宿主保护盾更重，说明该通路具有保护性，应避免使用QQ。",
        "reasoning": "生物学推理",
        "reason_harm": "**有效Gamma过高：** PI3K/AKT通路充当保护盾。QQ会移除该保护盾，增加转移风险。",
        "reason_prot": "**有效Gamma较低：** PI3K/AKT通路正在驱动转移。QQ将阻断AI-2信号，减少肿瘤负荷。",
        "save_btn": "💾 保存至患者历史", "saved": "已保存！",
        "sandbox_title": "🎮 治疗模拟沙盒：实时假设分析",
        "sandbox_exp": "<b>如何使用：</b> 调整滑块以模拟假想患者。<br><b>上图：</b> 显示QQ药物如何抑制细菌载量。<br><b>下图：</b> 显示对转移的下游影响。如果彩色线条上升，说明QQ有害；如果下降，说明QQ具有保护作用。",
        "sb_qq": "调整QQ剂量", "sb_gamma": "调整Gamma水平",
        "graph_bac_title": "细菌载量动态", "graph_met_title": "转移负荷",
        "x_axis_time": "时间（天）", "y_axis_abund": "相对丰度", "y_axis_met": "转移负荷",
        "legend_bac_no": "具核梭杆菌（无QQ）", "legend_bac_yes": "具核梭杆菌（QQ治疗）",
        "legend_met_no": "转移（无QQ）", "legend_met_yes": "转移（QQ治疗）",
        "sb_result_harm": "⚠️ **结果：** 当Gamma = {:.2f}（高于阈值{:.2f}）时，QQ治疗**有害**，因为它移除了保护通路。",
        "sb_result_prot": "✅ **结果：** 当Gamma = {:.2f}（低于阈值{:.2f}）时，QQ治疗**具有保护作用**，因为它阻断了促转移信号。",
        "xai_title": "🧠 AI可解释性 (XAI)",
        "xai_exp": "<b>这是什么？</b> 该瀑布图解释了AI的“大脑”。它展示了每个因素如何推高风险（红色）或降低风险（绿色）。",
        "xai_xaxis": "特征对风险的贡献",
        "colon_title": "🫁 3D人体结肠：分子通路可视化",
        "colon_exp": "<b>这是什么？</b> 人体结肠的3D全息视图。在组织内部，您可以看到生物通路：<b>具核梭杆菌</b>（橙色）、<b>毒力因子</b>（黄色）、<b>PI3K</b>（蓝色）、<b>AKT/Gamma</b>（绿色/红色）以及<b>转移</b>（绿色/红色）。用鼠标旋转图形进行探索。",
        "colon_tip": "💡 **提示：** 使用鼠标**旋转、缩放和平移**此3D模型。该通路悬浮在结肠组织内部。",
        "compare_title": "📊 临床试验模式：并排对比",
        "compare_exp": "<b>如何使用：</b> 在标签1中保存至少2名患者。此标签将显示散点图、对比表和汇总统计。",
        "compare_warn": "请至少保存2名患者至历史记录（标签1）以进行对比。",
        "compare_scatter": "### 患者聚类（Gamma vs. 具核梭杆菌载量）",
        "compare_table": "### 并排对比表",
        "compare_stats": "### 队列汇总统计",
        "compare_bar": "### 患者间风险对比",
        "report_title": "📝 智能报告",
        "report_exp": "<b>自动生成的临床笔记。</b>",
        "report_area": "生成的临床笔记：",
        "report_dl": "📄 下载笔记"
    }
}

# --- LANGUAGE SELECTOR ---
st.sidebar.header("🌐 " + translations["English"]["lang_select"] + " / " + translations["中文 (Chinese)"]["lang_select"])
lang = st.sidebar.selectbox("Language / 语言", ["English", "中文 (Chinese)"])
t = translations[lang]

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- TITLE ---
st.title(t["title"])
st.markdown(f"#### {t['subtitle']}")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header(t["sidebar_header"])
sex = st.sidebar.selectbox(t["sex"], [t["male"], t["female"]])
gamma_baseline = st.sidebar.slider(t["gamma"], 0.1, 2.0, 0.5, 0.1)
pik3ca = st.sidebar.selectbox(t["mutation"], [t["wildtype"], t["mutated"]])
f_nucleatum_load = st.sidebar.slider(t["load"], 0.0, 1.0, 0.3, 0.05)
qq_dose = st.sidebar.slider(t["dose"], 0.0, 1.5, 0.8, 0.1)

# --- MODEL LOGIC ---
bacterial_push = f_nucleatum_load * 0.5
gamma_effective = gamma_baseline + bacterial_push

threshold = 1.0
if sex == t["female"]: threshold = 0.8
elif sex == t["male"]: threshold = 1.2
if pik3ca == t["mutated"]: threshold -= 0.1

if gamma_effective > threshold:
    outcome = "HARMFUL"
    risk_pct = min(95, int((gamma_effective - threshold) * 100 + 50))
    color = "#ef4444"
    icon = "❌"
    rec = t["no_qq"]
else:
    outcome = "PROTECTIVE"
    risk_pct = min(95, int((threshold - gamma_effective) * 100 + 50))
    color = "#10b981"
    icon = "✅"
    rec = t["give_qq"]

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([t["tab1"], t["tab2"], t["tab3"], t["tab4"], t["tab5"], t["tab6"]])

# ==========================================
# TAB 1: RECOMMENDATION
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(t["diag_res"])
        if outcome == "HARMFUL": st.error(f"# {icon} {rec}")
        else: st.success(f"# {icon} {rec}")
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{risk_pct}%</div><div class="metric-label">{t["adv_risk"]}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{gamma_effective:.2f}</div><div class="metric-label">{t["eff_gamma"]}</div></div>', unsafe_allow_html=True)
        
        st.markdown(f"**{t['balance_title']}**")
        st.markdown(f"""<div class="explain-box">{t['balance_exp']}</div>""", unsafe_allow_html=True)
        
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
        st.subheader(t["reasoning"])
        if outcome == "HARMFUL": st.info(t["reason_harm"])
        else: st.info(t["reason_prot"])
        if st.button(t["save_btn"]):
            st.session_state.history.append({"ID": len(st.session_state.history)+1, "Sex": sex, "PIK3CA": pik3ca, "Gamma": round(gamma_effective, 2), "F. nucleatum": round(f_nucleatum_load, 2), "Outcome": outcome, "Risk": f"{risk_pct}%"})
            st.success(t["saved"])

# ==========================================
# TAB 2: SANDBOX
# ==========================================
with tab2:
    st.subheader(t["sandbox_title"])
    st.markdown(f"""<div class="explain-box">{t['sandbox_exp']}</div>""", unsafe_allow_html=True)
    
    sandbox_qq = st.slider(t["sb_qq"], 0.0, 1.5, qq_dose, 0.1)
    sandbox_gamma = st.slider(t["sb_gamma"], 0.1, 2.0, gamma_effective, 0.1)
    
    time_arr = np.linspace(0, 100, 100)
    b_growth = 1 / (1 + np.exp(-0.1 * (time_arr - 30))) * f_nucleatum_load
    b_qq = b_growth * (1 - 0.9 * sandbox_qq)
    
    fig_bacteria = go.Figure()
    fig_bacteria.add_trace(go.Scatter(x=time_arr, y=b_growth, mode='lines', name=t["legend_bac_no"], line=dict(color='orange', dash='dash', width=2)))
    fig_bacteria.add_trace(go.Scatter(x=time_arr, y=b_qq, mode='lines', name=t["legend_bac_yes"], line=dict(color='purple', width=4)))
    fig_bacteria.update_layout(title=t["graph_bac_title"], xaxis_title=t["x_axis_time"], yaxis_title=t["y_axis_abund"], template="plotly_white", height=350)
    st.plotly_chart(fig_bacteria, use_container_width=True)
    
    control_metastasis = 1 / (1 + np.exp(-0.1 * (time_arr - 50))) 
    if sandbox_gamma > threshold:
        sandbox_metastasis = control_metastasis * (1 + 0.5 * sandbox_qq * (sandbox_gamma - threshold)); sandbox_color = "#ef4444"
    else:
        sandbox_metastasis = control_metastasis * (1 - 0.8 * sandbox_qq); sandbox_color = "#10b981"
    
    fig_metastasis = go.Figure()
    fig_metastasis.add_trace(go.Scatter(x=time_arr, y=control_metastasis, mode='lines', name=t["legend_met_no"], line=dict(color='gray', dash='dash', width=2)))
    fig_metastasis.add_trace(go.Scatter(x=time_arr, y=sandbox_metastasis, mode='lines', name=t["legend_met_yes"], line=dict(color=sandbox_color, width=4)))
    fig_metastasis.update_layout(title=t["graph_met_title"], xaxis_title=t["x_axis_time"], yaxis_title=t["y_axis_met"], template="plotly_white", height=350)
    st.plotly_chart(fig_metastasis, use_container_width=True)
    
    if sandbox_gamma > threshold: st.error(t["sb_result_harm"].format(sandbox_gamma, threshold))
    else: st.success(t["sb_result_prot"].format(sandbox_gamma, threshold))

# ==========================================
# TAB 3: XAI
# ==========================================
with tab3:
    st.subheader(t["xai_title"])
    st.markdown(f"""<div class="explain-box">{t['xai_exp']}</div>""", unsafe_allow_html=True)
    
    mutation_label = f"PIK3CA ({pik3ca})"
    mutation_value = 15 if pik3ca == t["mutated"] else 0
    shap_values = {"F. nucleatum Load": f_nucleatum_load * 30, mutation_label: mutation_value, "Baseline Gamma": gamma_baseline * 10, f"Sex ({sex})": 10 if sex == t["female"] else -5, "QQ Dose Effect": -20 if outcome == "PROTECTIVE" else 20}
    fig_shap = go.Figure(go.Waterfall(
        name="SHAP", orientation="v", measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=list(shap_values.keys()) + ["Total Risk"], textposition="outside", text=[f"{v:+.1f}%" for v in shap_values.values()] + [f"{risk_pct}%"],
        y=list(shap_values.values()) + [risk_pct], connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"#10b981"}}, increasing={"marker":{"color":"#ef4444"}}, totals={"marker":{"color":"#0ea5e9"}}
    ))
    fig_shap.update_layout(title=t["xai_xaxis"], template="plotly_white", waterfallgap=0.3, height=500)
    st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# TAB 4: 3D HUMAN COLON
# ==========================================
with tab4:
    st.subheader(t["colon_title"])
    st.markdown(f"""<div class="explain-box">{t['colon_exp']}</div>""", unsafe_allow_html=True)
    
    fig_colon = go.Figure()
    t_arr = np.linspace(0, 4*np.pi, 100)
    x_tube = 10 * np.cos(t_arr); y_tube = 10 * np.sin(t_arr); z_tube = np.linspace(-5, 5, 100)
    
    fig_colon.add_trace(go.Mesh3d(x=x_tube, y=y_tube, z=z_tube, alphahull=0, opacity=0.15, color='#fda4af', name='Colon Tissue'))
    
    nodes = [
        {"x": 10, "y": 0, "z": -3, "text": f"F. nucleatum<br>Load: {f_nucleatum_load:.2f}", "color": "#f59e0b", "size": 15},
        {"x": 0, "y": 10, "z": -1.5, "text": "Fap2 / RadD<br>(Virulence)", "color": "#f97316", "size": 12},
        {"x": -10, "y": 0, "z": 0, "text": "PI3K<br>(Activation)", "color": "#3b82f6", "size": 12},
        {"x": 0, "y": -10, "z": 1.5, "text": f"AKT (Gamma)<br>{gamma_effective:.2f}", "color": "#10b981" if outcome == "PROTECTIVE" else "#ef4444", "size": 18},
        {"x": 10, "y": 0, "z": 3, "text": f"Metastasis<br>{rec}", "color": "#10b981" if outcome == "PROTECTIVE" else "#ef4444", "size": 15}
    ]
    
    for i in range(len(nodes)-1):
        fig_colon.add_trace(go.Scatter3d(x=[nodes[i]["x"], nodes[i+1]["x"]], y=[nodes[i]["y"], nodes[i+1]["y"]], z=[nodes[i]["z"], nodes[i+1]["z"]], mode='lines', line=dict(color='#64748b', width=5), showlegend=False))
    
    for n in nodes:
        fig_colon.add_trace(go.Scatter3d(x=[n["x"]], y=[n["y"]], z=[n["z"]], mode='markers+text', marker=dict(size=n["size"], color=n["color"], opacity=0.9, line=dict(color='white', width=2)), text=[n["text"]], textposition="top center", textfont=dict(size=11, color="black"), showlegend=False))
    
    fig_colon.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False, title=''), yaxis=dict(showbackground=False, showticklabels=False, title=''), zaxis=dict(showbackground=False, showticklabels=False, title=''), bgcolor="#f8fafc", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))), paper_bgcolor="#f8fafc", margin=dict(l=0, r=0, b=0, t=0), height=600)
    st.plotly_chart(fig_colon, use_container_width=True)
    st.info(t["colon_tip"])

# ==========================================
# TAB 5: COMPARE PATIENTS
# ==========================================
with tab5:
    st.subheader(t["compare_title"])
    st.markdown(f"""<div class="explain-box">{t['compare_exp']}</div>""", unsafe_allow_html=True)
    
    if len(st.session_state.history) < 2:
        st.info(t["compare_warn"])
    else:
        df = pd.DataFrame(st.session_state.history)
        
        st.markdown(t["compare_scatter"])
        fig_comp = go.Figure()
        for i, row in df.iterrows():
            fig_comp.add_trace(go.Scatter(x=[row['Gamma']], y=[row['F. nucleatum']], mode='markers+text', marker=dict(size=25, color='#ef4444' if row['Outcome'] == 'HARMFUL' else '#10b981', line=dict(color='black', width=2)), text=[f"Patient {row['ID']}"], textposition="top center"))
        fig_comp.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold:.2f})")
        fig_comp.update_layout(title="Patient Clustering", template="plotly_white", height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown(t["compare_table"])
        comparison_df = df[['ID', 'Sex', 'PIK3CA', 'Gamma', 'F. nucleatum', 'Outcome', 'Risk']].copy()
        comparison_df.columns = ['Patient ID', 'Sex', 'PIK3CA Status', 'Gamma', 'F. nucleatum Load', 'QQ Outcome', 'Adverse Risk'] if lang == "English" else ['患者ID', '性别', 'PIK3CA状态', 'Gamma', '具核梭杆菌载量', 'QQ结果', '不良风险']
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown(t["compare_stats"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients" if lang=="English" else "总患者数", len(df))
        c2.metric("Harmful (❌)" if lang=="English" else "有害 (❌)", len(df[df['Outcome'] == 'HARMFUL']))
        c3.metric("Protective (✅)" if lang=="English" else "保护 (✅)", len(df[df['Outcome'] == 'PROTECTIVE']))
        c4.metric("Avg Gamma" if lang=="English" else "平均Gamma", f"{df['Gamma'].mean():.2f}")
        
        st.markdown(t["compare_bar"])
        fig_bar = go.Figure(data=[
            go.Bar(x=[f"Patient {row['ID']}" if lang=="English" else f"患者 {row['ID']}" for _, row in df.iterrows()], y=[int(row['Risk'].replace('%', '')) for _, row in df.iterrows()], marker_color=['#ef4444' if row['Outcome'] == 'HARMFUL' else '#10b981' for _, row in df.iterrows()], text=[row['Risk'] for _, row in df.iterrows()], textposition='auto')
        ])
        fig_bar.update_layout(title="Adverse Risk per Patient" if lang=="English" else "每位患者的不良风险", xaxis_title="Patient" if lang=="English" else "患者", yaxis_title="Risk (%)" if lang=="English" else "风险 (%)", template="plotly_white", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 6: SMART REPORT
# ==========================================
with tab6:
    st.subheader(t["report_title"])
    st.markdown(f"""<div class="explain-box">{t['report_exp']}</div>""", unsafe_allow_html=True)
    if lang == "English":
        report = f"Patient: {sex}, PIK3CA {pik3ca}. Effective Gamma: {gamma_effective:.2f}. F. nucleatum Load: {f_nucleatum_load:.2f}.\n\nAI RECOMMENDATION: {rec} (Risk: {risk_pct}%)."
    else:
        report = f"患者：{sex}，PIK3CA {pik3ca}。有效Gamma：{gamma_effective:.2f}。具核梭杆菌载量：{f_nucleatum_load:.2f}。\n\nAI建议：{rec}（风险：{risk_pct}%）。"
    st.text_area(t["report_area"], report, height=200)
    st.download_button(t["report_dl"], data=report, file_name="Clinical_Note.txt")

# --- FOOTER ---
st.markdown("---")
st.caption("QuorumQuench AI | Master Version | Developed by Muhammad Imad Khan Khalil")
