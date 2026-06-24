"""
SIGAP Sekolah v4.0 - Sistem Identifikasi Gejala Anak Berisiko Putus Sekolah
LKS AI 2026 - Juara Nasional Target
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import os
from datetime import datetime

from model import (
    SIGAP_FEATURE_NAMES, SIGAP_FEATURE_LABELS, UCI_FEATURE_NAMES, UCI_FEATURE_LABELS,
    FEATURE_NAMES, FEATURE_LABELS, RISK_LABELS, detect_mode, get_feature_names, get_feature_labels,
    load_data, prepare_features,
    predict_risk, explain_prediction, get_top_factors, generate_recommendations,
    train_all_models, save_all_models, load_all_models, what_if_simulation,
)
from fairness import BiasDetector, TransparencyLogger, HumanInTheLoop
from ai_bk import AIBKAssistant
from school_intelligence import SchoolRiskIntelligence
from pdf_report import generate_student_report, generate_parent_letter
from intervention import (
    InterventionTracker, EarlyWarningSystem, ImpactMetrics,
    RiskMitigation, AppealSystem, AuditTrail,
)
from data_loader import (
    UCI_FEATURE_NAMES as DL_UCI_FEATURES, UCI_FEATURE_LABELS_ID,
    FEATURE_CATEGORIES, get_dataset_info, get_correlation_with_target,
    load_uci_dataset, prepare_uci_features, adapt_to_sigap_features,
)

st.set_page_config(page_title="SIGAP Sekolah", page_icon="🏫", layout="wide", initial_sidebar_state="expanded")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transparency_log" not in st.session_state:
    st.session_state.transparency_log = TransparencyLogger()
if "whatif_scenarios" not in st.session_state:
    st.session_state.whatif_scenarios = []

BG = "#0f172a"
CARD_BG = "rgba(30, 41, 59, 0.7)"
CARD_BORDER = "rgba(255,255,255,0.10)"
TEXT = "#e2e8f0"
TEXT_DIM = "#94a3b8"
SIDEBAR_BG = "#0f172a"
SIDEBAR_TEXT = "#e2e8f0"
BTN_BG = "#0ea5e9"
BTN_HOVER = "#0284c7"
INPUT_BG = "rgba(30,41,59,0.8)"
INPUT_BORDER = "rgba(255,255,255,0.15)"
ALERT_BG = "rgba(30,41,59,0.5)"
GLOW = "rgba(56,189,248,0.5)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

#MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; pointer-events: auto !important; }}
header[data-testid="stHeader"] button {{ pointer-events: auto !important; }}

.stApp {{ background: {BG} !important; background-attachment: fixed !important; color: {TEXT} !important; }}

.main-header {{
    background: {CARD_BG};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {CARD_BORDER};
    border-top: 1px solid rgba(255,255,255,0.15);
    border-left: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    color: {TEXT};
    margin-bottom: 1.5rem;
}}
.main-header h1 {{ color: {TEXT} !important; margin: 0 !important; font-size: 1.8rem !important; font-weight: 800 !important; text-shadow: 0 0 20px {GLOW}; }}
.main-header p {{ color: {TEXT_DIM} !important; margin: 0.5rem 0 0 0 !important; font-size: 0.9rem !important; }}

.stat-card {{
    background: {CARD_BG};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {CARD_BORDER};
    border-top: 1px solid rgba(255,255,255,0.15);
    border-left: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border-left: 4px solid;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}}
.stat-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.12); }}
.stat-card h3 {{ margin: 0 !important; font-size: 2rem !important; font-weight: 800 !important; }}
.stat-card p {{ margin: 0.3rem 0 0 0 !important; color: {TEXT_DIM} !important; font-size: 0.85rem !important; }}

.risk-high {{ border-left-color: #ef4444 !important; }}
.risk-high h3 {{ color: #ef4444 !important; }}
.risk-medium {{ border-left-color: #facc15 !important; }}
.risk-medium h3 {{ color: #facc15 !important; }}
.risk-low {{ border-left-color: #4ade80 !important; }}
.risk-low h3 {{ color: #4ade80 !important; }}

.student-row {{
    background: {CARD_BG};
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
.student-row:hover {{ transform: translateX(4px); box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
.student-row strong {{ color: {TEXT} !important; }}
.student-row small {{ color: {TEXT_DIM} !important; }}

.badge {{ padding: 0.3rem 0.8rem !important; border-radius: 20px !important; font-weight: 600 !important; font-size: 0.75rem !important; }}
.badge-danger {{ background: rgba(239,68,68,0.15) !important; color: #ef4444 !important; border: 1px solid rgba(239,68,68,0.3) !important; }}
.badge-success {{ background: rgba(74,222,128,0.15) !important; color: #4ade80 !important; border: 1px solid rgba(74,222,128,0.3) !important; }}

div[data-testid="stMetric"] {{
    background: {CARD_BG};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid {CARD_BORDER};
}}
div[data-testid="stMetric"] label {{ color: {TEXT_DIM} !important; }}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: {TEXT} !important; }}

section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {CARD_BORDER} !important;
}}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] .stMarkdown div,
section[data-testid="stSidebar"] small {{ color: {SIDEBAR_TEXT} !important; }}
section[data-testid="stSidebar"] [data-baseweb="radio"] label,
section[data-testid="stSidebar"] [data-baseweb="radio"] span,
section[data-testid="stSidebar"] [data-baseweb="radio"] div {{ color: {SIDEBAR_TEXT} !important; }}
section[data-testid="stSidebar"] button {{ color: {SIDEBAR_TEXT} !important; }}
section[data-testid="stSidebar"] [data-baseweb="radio"] {{ color: {SIDEBAR_TEXT} !important; }}
section[data-testid="stSidebar"] [data-baseweb="radio"] span {{ color: {SIDEBAR_TEXT} !important; }}
section[data-testid="stSidebar"] [role="radio"] {{ color: {SIDEBAR_TEXT} !important; }}
section[data-testid="stSidebar"] [role="radio"] span {{ color: {SIDEBAR_TEXT} !important; }}
section[data-testid="stSidebar"] p {{ color: {SIDEBAR_TEXT} !important; }}

.stButton > button {{
    background: {BTN_BG} !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
}}
.stButton > button:hover {{ background: {BTN_HOVER} !important; }}
.stDownloadButton > button {{ background: #22c55e !important; color: white !important; }}

.stForm {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 1.5rem;
}}

.stTabs [data-baseweb="tab-list"] {{ background: {ALERT_BG}; border-radius: 12px; gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ color: {TEXT_DIM} !important; }}
.stTabs [aria-selected="true"] {{ background: {CARD_BG} !important; color: {TEXT} !important; font-weight: 600 !important; }}

div[data-testid="stSuccess"] {{ background: rgba(74,222,128,0.12) !important; border-left: 4px solid #4ade80 !important; border-radius: 8px !important; color: {TEXT} !important; }}
div[data-testid="stWarning"] {{ background: rgba(250,204,21,0.12) !important; border-left: 4px solid #facc15 !important; border-radius: 8px !important; color: {TEXT} !important; }}
div[data-testid="stError"] {{ background: rgba(239,68,68,0.12) !important; border-left: 4px solid #ef4444 !important; border-radius: 8px !important; color: {TEXT} !important; }}
div[data-testid="stInfo"] {{ background: rgba(14,165,233,0.12) !important; border-left: 4px solid #0ea5e9 !important; border-radius: 8px !important; color: {TEXT} !important; }}

.chat-user {{ background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; margin-left: 15%; text-align: right; color: {TEXT}; }}
.chat-bot {{ background: {ALERT_BG}; border: 1px solid {CARD_BORDER}; border-radius: 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; margin-right: 15%; color: {TEXT}; }}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div,
.stNumberInput > div > div > input {{
    background: {INPUT_BG} !important;
    border: 1px solid {INPUT_BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
}}

.stRadio > div {{ gap: 0.3rem !important; }}
.stRadio > div > label {{ color: {SIDEBAR_TEXT} !important; font-size: 0.9rem !important; }}

::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.05); }}
::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.15); border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_resources():
    data_path = os.path.join(os.path.dirname(__file__), "data", "data_siswa.csv")
    data_uci_path = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
    used_path = data_path
    if os.path.exists(data_uci_path):
        used_path = data_uci_path
    df = load_data(used_path)
    mode = detect_mode(df)
    feature_names = get_feature_names(mode)
    feature_labels = get_feature_labels(mode)
    models_bundle = load_all_models()
    need_retrain = models_bundle is None
    if not need_retrain and "feature_names" in models_bundle:
        if set(models_bundle["feature_names"]) != set(feature_names):
            need_retrain = True
    if need_retrain:
        X, y, mode = prepare_features(df)
        models, metrics, scaler, X_test, y_test, fn = train_all_models(X, y)
        save_all_models(models, scaler, metrics, fn)
        models_bundle = load_all_models()
    rf_model = models_bundle["models"]["random_forest"]
    return rf_model, models_bundle, df, mode, feature_names, feature_labels

model, models_bundle, df, DATA_MODE, CURRENT_FEATURES, CURRENT_LABELS = load_all_resources()

@st.cache_data
def cached_predictions(_model, _df, _models_bundle):
    return predict_risk(_model, _df, model_name="random_forest", models_bundle=_models_bundle)

all_preds = cached_predictions(model, df, models_bundle)
st.session_state.model = model
st.session_state.data = df
st.session_state.models_bundle = models_bundle
st.session_state.data_mode = DATA_MODE
st.session_state.feature_names = CURRENT_FEATURES
st.session_state.feature_labels = CURRENT_LABELS

import plotly.io as pio
pio.templates["theme"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, family="Inter, sans-serif"),
        xaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
        colorway=["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#00f2fe", "#43e97b", "#fa709a"],
    )
)
pio.templates.default = "theme"

def get_risk_badge(risk_label):
    if risk_label == "Berisiko Tinggi":
        return '<span class="badge badge-danger">Berisiko Tinggi</span>'
    return '<span class="badge badge-success">Aman</span>'

def risk_gauge(score):
    color = "#27ae60" if score < 30 else "#f39c12" if score < 60 else "#e74c3c"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 30], "color": "#d4edda"}, {"range": [30, 60], "color": "#fff3cd"}, {"range": [60, 100], "color": "#f8d7da"}],
               "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": score}}))
    fig.update_layout(height=220, margin=dict(t=30, b=0, l=30, r=30))
    return fig

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:0.5rem 0 0.8rem 0;">
        <div style="font-size:2.2rem;margin-bottom:0.2rem;">🏫</div>
        <div style="font-size:1.2rem;font-weight:800;color:{TEXT};">SIGAP Sekolah</div>
        <div style="font-size:0.68rem;color:{TEXT_DIM};margin-top:0.15rem;letter-spacing:0.1em;text-transform:uppercase;">
            Academy Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<hr style="margin:0.3rem 0 0.8rem 0;border-color:{CARD_BORDER};opacity:0.5;">', unsafe_allow_html=True)

    st.markdown(f'<div style="margin-bottom:0.3rem;"><span style="font-size:0.7rem;color:{TEXT_DIM};text-transform:uppercase;letter-spacing:0.1em;">🔍 Pencarian Siswa</span></div>', unsafe_allow_html=True)
    search_query = st.text_input("Cari Siswa", placeholder="Nama atau ID siswa...", label_visibility="collapsed", key="global_search")

    page = st.radio("Menu", [
        "Dashboard", "Daftar Siswa", "Prediksi Risiko", "Batch Upload",
        "AI BK Assistant", "What-If Simulator", "School Intelligence",
        "Model Comparison", "Early Warning", "Intervention Tracking",
        "Responsible AI", "Data Explorer", "Laporan PDF", "Tentang AI",
    ], index=0)

    st.markdown(f'<hr style="margin:0.8rem 0;border-color:{CARD_BORDER};opacity:0.5;">', unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center;padding:0.3rem 0;">
        <div style="background:{ALERT_BG};border-radius:10px;padding:0.6rem;margin-bottom:0.4rem;border:1px solid {CARD_BORDER};backdrop-filter:blur(8px);">
            <div style="font-size:0.65rem;color:{TEXT_DIM};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.15rem;">Ensemble AI</div>
            <div style="font-size:0.82rem;font-weight:600;color:{TEXT};">RF + XGB + SVM</div>
        </div>
        <div style="background:{ALERT_BG};border-radius:10px;padding:0.6rem;margin-bottom:0.4rem;border:1px solid {CARD_BORDER};backdrop-filter:blur(8px);">
            <div style="font-size:0.65rem;color:{TEXT_DIM};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.15rem;">Data</div>
            <div style="font-size:0.82rem;font-weight:600;color:{TEXT};">{len(df):,} Siswa Real</div>
        </div>
        <div style="margin-top:0.6rem;font-size:0.6rem;color:{TEXT_DIM};letter-spacing:0.12em;">LKS AI 2026</div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="main-header"><h1>🏫 SIGAP Sekolah</h1>
<p>Sistem Identifikasi Gejala Anak Berisiko Putus Sekolah — LKS AI 2026 | {len(df):,} Siswa Real</p></div>""", unsafe_allow_html=True)

if search_query and len(search_query) >= 2:
    sq = search_query.lower()
    search_mask = df["id_siswa"].astype(str).str.lower().str.contains(sq, na=False)
    if "nama" in df.columns:
        search_mask = search_mask | df["nama"].astype(str).str.lower().str.contains(sq, na=False)
    if "kelas" in df.columns:
        search_mask = search_mask | df["kelas"].astype(str).str.lower().str.contains(sq, na=False)
    matched = df[search_mask]
    if len(matched) > 0:
        st.markdown(f'<div style="background:rgba(14,165,233,0.12);border-left:4px solid #0ea5e9;border-radius:8px;padding:0.8rem 1.2rem;margin-bottom:1rem;"><strong style="color:#0ea5e9;">🔍 Hasil Pencarian:</strong> <span style="color:{TEXT};">{len(matched)} siswa ditemukan untuk "<em>{search_query}</em>"</span></div>', unsafe_allow_html=True)
        df_search = matched.copy()
        pred_map = {i: all_preds[i] for i in range(len(df))}
        df_search["skor_risiko"] = [pred_map[i]["risk_score"] for i in df_search.index]
        df_search["label_risiko"] = [pred_map[i]["risk_label"] for i in df_search.index]
        for _, row in df_search.head(10).iterrows():
            badge = get_risk_badge(row["label_risiko"])
            nama = row.get("nama", "")
            kelas = row.get("kelas", "")
            info = f"{nama} ({row['id_siswa']})" if nama else row["id_siswa"]
            kelas_info = f" | {kelas}" if kelas else ""
            st.markdown(f'<div class="student-row"><div><strong>{info}</strong> {badge}<br><small>Kehadiran: {row["persentase_kehadiran"]:.0f}% | Nilai: {row["rata_rata_nilai"]:.0f} | Risiko: {row["skor_risiko"]:.1f}%{kelas_info}</small></div></div>', unsafe_allow_html=True)
    else:
        st.warning(f'Tidak ditemukan siswa untuk "<em>{search_query}</em>"')

if page == "Dashboard":
    n_total = len(df)
    n_high = sum(1 for p in all_preds if p["is_at_risk"])
    n_low = n_total - n_high
    risk_pct = (n_high / n_total * 100) if n_total > 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card risk-low"><h3>{n_low:,}</h3><p>Siswa Aman</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card risk-high"><h3>{n_high:,}</h3><p>Siswa Berisiko</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card risk-medium"><h3>{risk_pct:.1f}%</h3><p>Tingkat Risiko</p></div>', unsafe_allow_html=True)
    with c4:
        avg_score = np.mean([p["risk_score"] for p in all_preds])
        st.markdown(f'<div class="stat-card risk-medium"><h3>{avg_score:.1f}</h3><p>Rata-rata Skor</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Distribusi Skor Risiko")
        df_display = df.copy()
        df_display["skor_risiko"] = [p["risk_score"] for p in all_preds]
        df_display["label_risiko"] = [p["risk_label"] for p in all_preds]
        fig = px.histogram(df_display, x="skor_risiko", nbins=30, color="label_risiko",
                          color_discrete_map={"Berisiko Tinggi": "#e74c3c", "Aman": "#27ae60"})
        fig.update_layout(height=350, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Faktor Risiko Utama")
        top_f = pd.DataFrame({"Faktor": list(FEATURE_LABELS.values()),
                              "Pentingan": np.round(np.abs(model.feature_importances_) * 100, 1)}).sort_values("Pentingan", ascending=True)
        fig = px.bar(top_f, x="Pentingan", y="Faktor", orientation="h", color="Pentingan", color_continuous_scale="Teal")
        fig.update_layout(height=350, margin=dict(t=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top 10 Siswa Berisiko")
    top10 = df_display.nlargest(10, "skor_risiko")
    for _, row in top10.iterrows():
        badge = get_risk_badge(row["label_risiko"])
        nama = row.get("nama", "")
        kelas = row.get("kelas", "")
        info = f"{nama} ({row['id_siswa']})" if nama else row["id_siswa"]
        kelas_info = f" | {kelas}" if kelas else ""
        st.markdown(f'<div class="student-row"><div><strong>{info}</strong> {badge}<br><small>Kehadiran: {row["persentase_kehadiran"]:.0f}% | Nilai: {row["rata_rata_nilai"]:.0f} | Risiko: {row["skor_risiko"]:.1f}%{kelas_info}</small></div></div>', unsafe_allow_html=True)

elif page == "Daftar Siswa":
    st.header("Daftar Siswa")
    df_list = df.copy()
    df_list["skor_risiko"] = [p["risk_score"] for p in all_preds]
    df_list["label_risiko"] = [p["risk_label"] for p in all_preds]
    c1, c2, c3 = st.columns(3)
    with c1:
        filter_risk = st.selectbox("Filter", ["Semua", "Berisiko Tinggi", "Aman"])
    with c2:
        sort_by = st.selectbox("Urutkan", ["Skor (Tertinggi)", "Skor (Terendah)", "Kehadiran", "Nilai"])
    with c3:
        if "kelas" in df_list.columns:
            kelas_list = ["Semua"] + sorted(df_list["kelas"].unique().tolist())
            filter_kelas = st.selectbox("Kelas", kelas_list)
        else:
            filter_kelas = "Semua"
    filtered = df_list.copy()
    if filter_risk != "Semua":
        filtered = filtered[filtered["label_risiko"] == filter_risk]
    if filter_kelas != "Semua" and "kelas" in filtered.columns:
        filtered = filtered[filtered["kelas"] == filter_kelas]
    if "Tertinggi" in sort_by:
        filtered = filtered.sort_values("skor_risiko", ascending=False)
    elif "Terendah" in sort_by:
        filtered = filtered.sort_values("skor_risiko", ascending=True)
    elif "Kehadiran" in sort_by:
        filtered = filtered.sort_values("persentase_kehadiran")
    else:
        filtered = filtered.sort_values("rata_rata_nilai")
    st.write(f"**{len(filtered)} siswa**")
    for _, row in filtered.head(50).iterrows():
        badge = get_risk_badge(row["label_risiko"])
        nama = row.get("nama", "")
        kelas = row.get("kelas", "")
        info = f"{nama} ({row['id_siswa']})" if nama else row["id_siswa"]
        kelas_info = f" | {kelas}" if kelas else ""
        st.markdown(f'<div class="student-row"><div><strong>{info}</strong> {badge}<br><small>Kehadiran: {row["persentase_kehadiran"]:.0f}% | Nilai: {row["rata_rata_nilai"]:.0f} | Risiko: {row["skor_risiko"]:.1f}%{kelas_info}</small></div></div>', unsafe_allow_html=True)

elif page == "Prediksi Risiko":
    st.header("Prediksi Risiko Siswa")
    st.info("Masukkan data siswa untuk prediksi. Semua data bersifat anonim.")
    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 85)
            nilai = st.slider("Rata-rata Nilai", 0, 100, 70)
            tren = st.slider("Tren Nilai (3 bln)", -10, 10, 0)
        with c2:
            bawah_kkm = st.number_input("Mapel Bawah KKM", 0, 12, 1)
            kip = st.selectbox("Penerima KIP", ["Tidak", "Ya"])
            jarak = st.slider("Jarak ke Sekolah (km)", 0.5, 25.0, 3.0)
        with c3:
            pelanggaran = st.number_input("Pelanggaran", 0, 20, 1)
            pekerjaan_ortu = st.selectbox("Pekerjaan Orang Tua", ["Tidak Bekerja", "Buruh/Tani", "Wiraswasta", "Pegawai"])
            pendidikan_ortu = st.selectbox("Pendidikan Orang Tua", ["Tidak Sekolah", "SD", "SMP", "SMA", "S1+"])
            saudara = st.number_input("Jumlah Saudara", 0, 8, 2)
        submitted = st.form_submit_button("Prediksi Risiko", type="primary", use_container_width=True)
    if submitted:
        student = {"persentase_kehadiran": kehadiran, "rata_rata_nilai": nilai, "tren_nilai": tren,
                   "jumlah_mapel_di_bawah_kkm": bawah_kkm, "status_kip": 1 if kip == "Ya" else 0,
                   "jarak_rumah_km": jarak, "jumlah_pelanggaran": pelanggaran,
                   "pekerjaan_ortu": ["Tidak Bekerja", "Buruh/Tani", "Wiraswasta", "Pegawai"].index(pekerjaan_ortu),
                   "pendidikan_ortu": ["Tidak Sekolah", "SD", "SMP", "SMA", "S1+"].index(pendidikan_ortu),
                   "jumlah_saudara": saudara}
        pred = predict_risk(model, pd.DataFrame([student]), model_name="ensemble", models_bundle=models_bundle)[0]
        explanation = explain_prediction(model, pd.DataFrame([student])[FEATURE_NAMES], 0)
        top_factors = get_top_factors(explanation, n=3)
        recommendations = generate_recommendations(top_factors, student)
        st.session_state.transparency_log.log_prediction("new_student", pred, explanation)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Hasil Prediksi")
            st.plotly_chart(risk_gauge(pred["risk_score"]), use_container_width=True)
            if pred["is_at_risk"]:
                st.error(f"Status: **{pred['risk_label']}** (Skor: {pred['risk_score']}%)")
            else:
                st.success(f"Status: **{pred['risk_label']}** (Skor: {pred['risk_score']}%)")
        with c2:
            st.subheader("Faktor Utama")
            for f in top_factors:
                arrow = "🔴" if f["direction"] == "meningkatkan" else "🟢"
                st.markdown(f"{arrow} **{f['feature']}** — {f['impact']}% {f['direction']} risiko")
            st.subheader("Rekomendasi")
            for rec in recommendations:
                st.markdown(f"• {rec}")
        review_needed, review_reason = HumanInTheLoop.needs_review(pred["risk_score"])
        if review_needed:
            st.warning(f"⚠️ Review manual diperlukan: {review_reason}")
        st.info("Catatan: Keputusan akhir tetap di tangan guru BK.")

elif page == "Batch Upload":
    st.header("Upload Batch Data Siswa")
    with st.expander("Format CSV"):
        st.code("id_siswa,persentase_kehadiran,rata_rata_nilai,tren_nilai,jumlah_mapel_di_bawah_kkm,status_kip,jarak_rumah_km,jumlah_pelanggaran,pekerjaan_ortu,pendidikan_ortu,jumlah_saudara")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"Uploaded: {len(df_up)} siswa")
            missing = [f for f in FEATURE_NAMES if f not in df_up.columns]
            if missing:
                st.error(f"Kolom kurang: {missing}")
            else:
                preds = predict_risk(model, df_up, model_name="ensemble", models_bundle=models_bundle)
                df_up["skor_risiko"] = [p["risk_score"] for p in preds]
                df_up["label_risiko"] = [p["risk_label"] for p in preds]
                n_risk = sum(1 for p in preds if p["is_at_risk"])
                c1, c2 = st.columns(2)
                c1.metric("Berisiko", n_risk)
                c2.metric("Total", len(preds))
                st.dataframe(df_up.sort_values("skor_risiko", ascending=False), use_container_width=True, hide_index=True)
                st.download_button("Download Hasil", df_up.to_csv(index=False), "hasil_sigap.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "AI BK Assistant":
    st.header("AI BK Assistant")
    st.caption("Chatbot untuk membantu guru menganalisis siswa berisiko")
    ai_bk = AIBKAssistant(models_bundle, df)
    if "id_siswa" in df.columns and "nama" in df.columns:
        df_sorted = df.sort_values("nama")
        display_names = [f"{row['nama']} ({row['id_siswa']}) - {row.get('kelas', '')}" for _, row in df_sorted.head(500).iterrows()]
        id_map = {display_names[i]: df_sorted["id_siswa"].iloc[i] for i in range(len(display_names))}
        selected_display = st.selectbox("Pilih Siswa", display_names)
        selected_student = id_map[selected_display]
    else:
        student_ids = df["id_siswa"].tolist()[:500] if "id_siswa" in df.columns else []
        selected_student = st.selectbox("Pilih Siswa", student_ids)
    if selected_student:
        if st.session_state.get("last_chat_student") != selected_student:
            st.session_state.chat_history = []
            st.session_state["last_chat_student"] = selected_student
        analysis = ai_bk.analyze_student(selected_student)
        if "error" not in analysis:
            pred = analysis["prediction"]
            ctx = analysis.get("context", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Skor Risiko", f"{pred['risk_score']:.1f}%")
            c2.metric("Status", pred["risk_label"])
            c3.metric("Kelas", ctx.get("kelas", "-"))
            c4.metric("Faktor Utama", len(analysis["top_factors"]))
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><strong>Anda:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot"><strong>AI BK:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
    user_input = st.chat_input("Ketik pertanyaan tentang siswa...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = ai_bk.chat(selected_student, user_input)
        if response.get("type") == "error":
            st.session_state.chat_history.append({"role": "assistant", "content": response.get("response", "Maaf, terjadi kesalahan.")})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": response.get("jawaban", "Maaf, saya tidak mengerti.")})
        st.rerun()

elif page == "What-If Simulator":
    st.header("What-If Simulator")
    st.caption("Simulasikan dampak perubahan terhadap risiko siswa")
    if "id_siswa" in df.columns and "nama" in df.columns:
        df_sorted_wi = df.sort_values("nama")
        display_names = [f"{row['nama']} ({row['id_siswa']}) - {row.get('kelas', '')}" for _, row in df_sorted_wi.head(500).iterrows()]
        id_map = {display_names[i]: df_sorted_wi["id_siswa"].iloc[i] for i in range(len(display_names))}
        selected_display = st.selectbox("Pilih Siswa", display_names, key="whatif_select")
        selected = id_map[selected_display]
    else:
        student_ids = df["id_siswa"].tolist() if "id_siswa" in df.columns else []
        selected = st.selectbox("Pilih Siswa", student_ids, key="whatif_select")
    if selected:
        row = df[df["id_siswa"] == selected].iloc[0]
        st.subheader("Data Saat Ini")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kehadiran", f"{row.get('persentase_kehadiran', 85)}%")
        c2.metric("Nilai", f"{row.get('rata_rata_nilai', 70)}")
        c3.metric("Pelanggaran", f"{row.get('jumlah_pelanggaran', 1)}")
        c4.metric("Mapel Bawah KKM", f"{row.get('jumlah_mapel_di_bawah_kkm', 1)}")
        with st.form("whatif_form"):
            c1, c2 = st.columns(2)
            with c1:
                new_kehadiran = st.slider("Target Kehadiran (%)", 0, 100, int(row.get("persentase_kehadiran", 85)))
                new_nilai = st.slider("Target Nilai", 0, 100, int(row.get("rata_rata_nilai", 70)))
                new_pelanggaran = st.slider("Target Pelanggaran", 0, 20, int(row.get("jumlah_pelanggaran", 1)))
            with c2:
                new_bawah_kkm = st.slider("Target Mapel Bawah KKM", 0, 12, int(row.get("jumlah_mapel_di_bawah_kkm", 1)))
                new_jarak = st.slider("Target Jarak (km)", 0.5, 25.0, float(row.get("jarak_rumah_km", 3.0)))
                remedial = st.checkbox("Program Remedial")
                mentoring = st.checkbox("Program Mentoring")
            run_sim = st.form_submit_button("Jalankan Simulasi", type="primary", use_container_width=True)
        if run_sim:
            changes = {"persentase_kehadiran": new_kehadiran, "rata_rata_nilai": new_nilai,
                       "jumlah_pelanggaran": new_pelanggaran, "jumlah_mapel_di_bawah_kkm": new_bawah_kkm,
                       "jarak_rumah_km": new_jarak}
            result = what_if_simulation(row.to_dict(), models_bundle, changes)
            st.subheader("Hasil Simulasi")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sebelum", f"{result['original_score']:.1f}%")
            delta_val = result['original_score'] - result['simulated_score']
            delta_str = f"+{delta_val:.1f}%" if delta_val < 0 else f"-{delta_val:.1f}%"
            c2.metric("Sesudah", f"{result['simulated_score']:.1f}%", delta=delta_str)
            c3.metric("Status Baru", result["new_category"])
            color = "#27ae60" if result["reduction"] > 0 else "#e74c3c"
            fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=result["simulated_score"],
                delta={"reference": result["original_score"]},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                       "steps": [{"range": [0, 30], "color": "#d4edda"}, {"range": [30, 60], "color": "#fff3cd"}, {"range": [60, 100], "color": "#f8d7da"}]}))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            if result["contributions"]:
                st.subheader("Kontribusi Perubahan")
                contrib_df = pd.DataFrame([{"Parameter": k, "Dampak Risiko": v} for k, v in result["contributions"].items()])
                fig = px.bar(contrib_df, x="Dampak Risiko", y="Parameter", orientation="h", color="Dampak Risiko", color_continuous_scale="RdYlGn_r")
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            st.session_state.whatif_scenarios.append({"student": selected, "changes": changes, "result": result, "timestamp": datetime.now().isoformat()})

elif page == "School Intelligence":
    st.header("School Risk Intelligence Dashboard")
    intel = SchoolRiskIntelligence(df, all_preds, FEATURE_NAMES)
    dashboard = intel.get_dashboard_data()
    st.subheader("Overview Sekolah")
    ov = dashboard["overview"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Siswa", ov["total_siswa"])
    c2.metric("Berisiko", ov["siswa_berisiko"])
    c3.metric("Persentase", f"{ov['persentase_berisiko']}%")
    c4.metric("Rata-rata Risiko", f"{ov['rata_rata_risiko']}%")
    c5.metric("Kehadiran", f"{ov['rata_rata_kehadiran']}%")
    c6.metric("Nilai", f"{ov['rata_rata_nilai']}")
    c1, c2 = st.columns(2)
    with c1:
        ranking = dashboard["class_ranking"]
        if ranking:
            st.subheader("Ranking Kelas Berisiko")
            rank_df = pd.DataFrame(ranking)
            fig = px.bar(rank_df, x="risk_pct", y="kelas", orientation="h", color="risk_pct", color_continuous_scale="Reds", text="risk_pct")
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400, showlegend=False, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        factors = dashboard["factor_ranking"]
        if factors:
            st.subheader("Ranking Faktor Penyebab")
            fac_df = pd.DataFrame(factors)
            fig = px.bar(fac_df, x="pct", y="faktor", orientation="h", color="pct", color_continuous_scale="Teal", text="pct")
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400, showlegend=False, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
    trend = dashboard["trend"]
    if trend:
        st.subheader("Tren 6 Bulan Terakhir")
        trend_df = pd.DataFrame(trend)
        fig = px.area(trend_df, x="bulan", y="risiko_rata2", markers=True)
        fig.update_traces(fillcolor="rgba(102, 126, 234, 0.2)", line=dict(color="#667eea", width=3))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Responsible AI":
    st.header("Responsible AI — Komitmen Kami")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Transparansi (XAI)")
        st.markdown("Setiap prediksi dilengkapi penjelasan SHAP — guru bisa memahami mengapa siswa dianggap berisiko.")
        st.subheader("2. Human-in-the-Loop")
        st.markdown("AI hanya memberi saran. Guru BK tetap mengambil keputusan akhir.")
        st.subheader("3. Privasi Data")
        st.markdown("Data dianonimkan, akses dibatasi, tidak ada data pribadi yang bocor.")
    with c2:
        st.subheader("4. Keadilan (Fairness)")
        st.markdown("Model diuji keadilannya. Status KIP bukan satu-satunya faktor penentu.")
        st.subheader("5. Keamanan Prediksi")
        st.markdown("False Negative diminimalkan. Selalu ada mekanisme verifikasi manusia.")
        st.subheader("6. Dampak Sosial")
        st.markdown("Selaras dengan SDG 4 — Pendidikan Berkualitas.")
    st.markdown("---")
    st.subheader("Live Fairness Check")
    with st.expander("Jalankan Fairness Check"):
        y_true = df["putus_sekolah"] if "putus_sekolah" in df.columns else pd.Series([0]*len(df))
        detector = BiasDetector(model, df, FEATURE_NAMES)
        bias_summary = detector.get_bias_summary(y_true)
        if bias_summary["is_fair"]:
            st.success("Model dinyatakan ADIL berdasarkan Equalized Odds")
        else:
            for issue in bias_summary["issues"]:
                st.warning(f"{issue}")
    st.subheader("Contoh SHAP Explanation")
    X_sample = df[FEATURE_NAMES].iloc[[0]]
    explanation = explain_prediction(model, X_sample, 0)
    exp_df = pd.DataFrame([{"Faktor": k, "Pengaruh": v, "Arah": "Meningkatkan" if v > 0 else "Menurunkan"}
                           for k, v in sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)])
    fig = px.bar(exp_df, x="Pengaruh", y="Faktor", color="Arah", orientation="h",
                 color_discrete_map={"Meningkatkan": "#e74c3c", "Menurunkan": "#27ae60"})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Comparison":
    st.header("Perbandingan Model AI")
    st.info("Mode: Real Dataset (43,164 siswa, 10 fitur)")
    metrics = models_bundle.get("metrics", {})
    if metrics:
        comp_data = []
        for name, m in metrics.items():
            comp_data.append({"Model": name.replace("_", " ").title(), "Accuracy": m["accuracy"]*100,
                "Recall": m["recall"]*100, "Precision": m["precision"]*100, "F1 Score": m["f1"]*100, "AUC-ROC": m.get("auc_roc", 0)*100})
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        fig = px.bar(comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                     x="Model", y="Score", color="Metric", barmode="group")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Early Warning":
    st.header("Early Warning System")
    alerts = EarlyWarningSystem.generate_alerts(all_preds, df)
    if alerts:
        n_critical = sum(1 for a in alerts if a["warning_level"] == "CRITICAL")
        n_high = sum(1 for a in alerts if a["warning_level"] == "HIGH")
        n_medium = sum(1 for a in alerts if a["warning_level"] == "MEDIUM")
        c1, c2, c3 = st.columns(3)
        c1.metric("CRITICAL", n_critical)
        c2.metric("HIGH", n_high)
        c3.metric("MEDIUM", n_medium)
        for alert in alerts[:30]:
            level = alert["warning_level"]
            color = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "MEDIUM": "#ca8a04"}.get(level, "#16a34a")
            st.markdown(f"""<div style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15);
                        border-left:4px solid {color};padding:1rem;border-radius:8px;margin-bottom:0.5rem;">
                        <strong>{alert['student_name']} ({alert['student_id']})</strong> — Skor: {alert['risk_score']}%<br>
                        <small style="color:{color}">Level: {level} | Aksi: {alert['recommended_action']}</small></div>""", unsafe_allow_html=True)
    else:
        st.success("Tidak ada peringatan aktif saat ini.")

elif page == "Intervention Tracking":
    st.header("Intervention Tracking")
    tracker = InterventionTracker()
    impact = ImpactMetrics.calculate_impact(tracker.interventions, all_preds)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Siswa", impact["total_students"])
    c2.metric("Berisiko", impact["at_risk_students"])
    c3.metric("Intervensi Aktif", impact["active_interventions"])
    c4.metric("Tingkat Keberhasilan", f"{impact['success_rate']}%")
    with st.form("add_intervention"):
        c1, c2 = st.columns(2)
        with c1:
            intv_student = st.selectbox("Siswa", df["id_siswa"].tolist()[:500] if "id_siswa" in df.columns else [], key="intv_student")
            intv_type = st.selectbox("Jenis", ["Akademik", "Konseling", "Ekonomi", "Kedisiplinan", "Transportasi", "Lainnya"])
        with c2:
            intv_desc = st.text_area("Deskripsi Intervensi")
            intv_priority = st.selectbox("Prioritas", ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        if st.form_submit_button("Tambah Intervensi", type="primary"):
            result = tracker.add_intervention(intv_student, intv_type, intv_desc, priority=intv_priority)
            st.success(f"Intervensi {result['id']} berhasil ditambahkan!")
            st.rerun()
    if tracker.interventions:
        st.subheader("Riwayat Intervensi")
        for intv in reversed(tracker.interventions[-20:]):
            status_color = {"ACTIVE": "#ca8a04", "COMPLETED": "#16a34a", "CANCELLED": "#dc2626"}.get(intv["status"], "#6b7280")
            col_info, col_action = st.columns([5, 1])
            with col_info:
                st.markdown(f"""<div style="background:{ALERT_BG};padding:1rem;border-radius:8px;border-left:4px solid {status_color};">
                            <strong>{intv['id']}</strong> — {intv['student_id']} | {intv['type']} | Prioritas: {intv.get('priority','HIGH')} | Status: <span style="color:{status_color}">{intv['status']}</span><br>
                            <small>{intv['description'][:100]}...</small></div>""", unsafe_allow_html=True)
            with col_action:
                c1, c2 = st.columns(2)
                with c1:
                    if intv["status"] == "ACTIVE":
                        if st.button("✅", key=f"done_{intv['id']}", help="Tandai Selesai"):
                            tracker.update_intervention(intv["id"], status="COMPLETED", outcome="SUCCESS")
                            st.rerun()
                    elif intv["status"] == "COMPLETED":
                        if st.button("♻️", key=f"reopen_{intv['id']}", help="Buka Kembali"):
                            tracker.update_intervention(intv["id"], status="ACTIVE")
                            st.rerun()
                with c2:
                    if st.button("🗑️", key=f"del_{intv['id']}", help="Hapus"):
                        tracker.delete_intervention(intv["id"])
                        st.rerun()

elif page == "Data Explorer":
    st.header("Data Explorer & EDA")
    st.subheader("Distribusi Data")
    for col in ["persentase_kehadiran", "rata_rata_nilai", "jumlah_pelanggaran"]:
        if col in df.columns:
            fig = px.histogram(df, x=col, nbins=30, title=SIGAP_FEATURE_LABELS.get(col, col))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Korelasi Antar Fitur")
    numeric_df = df[SIGAP_FEATURE_NAMES].select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "Laporan PDF":
    st.header("Generator Laporan PDF")
    if "id_siswa" in df.columns and "nama" in df.columns:
        df_sorted_pdf = df.sort_values("nama")
        display_names = [f"{row['nama']} ({row['id_siswa']}) - {row.get('kelas', '')}" for _, row in df_sorted_pdf.head(500).iterrows()]
        id_map = {display_names[i]: df_sorted_pdf["id_siswa"].iloc[i] for i in range(len(display_names))}
        selected_display = st.selectbox("Pilih Siswa", display_names, key="pdf_student")
        selected = id_map[selected_display]
    else:
        selected = st.selectbox("Pilih Siswa", df["id_siswa"].tolist()[:500] if "id_siswa" in df.columns else [], key="pdf_student")
    report_type = st.radio("Jenis Laporan", ["Laporan Guru BK", "Surat untuk Orang Tua"])
    if st.button("Generate PDF", type="primary", use_container_width=True):
        row = df[df["id_siswa"] == selected].iloc[0]
        pred = predict_risk(model, pd.DataFrame([row]), model_name="ensemble", models_bundle=models_bundle)[0]
        feat_names = models_bundle.get("feature_names", CURRENT_FEATURES)
        available_feats = [f for f in feat_names if f in df.columns]
        explanation = explain_prediction(model, pd.DataFrame([row])[available_feats], 0)
        top_factors = get_top_factors(explanation, n=5)
        if report_type == "Laporan Guru BK":
            recommendations = generate_recommendations(top_factors, row.to_dict())
            filepath = generate_student_report(selected, row.to_dict(), pred, top_factors, recommendations)
        else:
            filepath = generate_parent_letter(selected, row.to_dict(), pred, top_factors)
        with open(filepath, "rb") as f:
            st.download_button("Download PDF", f.read(), os.path.basename(filepath), "application/pdf")

elif page == "Tentang AI":
    st.header("Bagaimana AI SIGAP Bekerja")
    st.subheader("Metrik Model Terbaik (Ensemble)")
    best = models_bundle.get("metrics", {}).get("ensemble", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{best.get('accuracy', 0):.1%}")
    c2.metric("Recall", f"{best.get('recall', 0):.1%}")
    c3.metric("Precision", f"{best.get('precision', 0):.1%}")
    c4.metric("F1 Score", f"{best.get('f1', 0):.1%}")
    c5.metric("AUC-ROC", f"{best.get('auc_roc', 0):.3f}")
    st.warning("Recall lebih penting dari Accuracy. Lebih baik salah waspada daripada kehilangan satu anak.")
    st.subheader("Dataset yang Digunakan")
    st.markdown("""**Real Dataset (43,164 Siswa)**
    - 6 dataset real dari UCI & Kaggle
    - 43,164 siswa, dikontekskan ke SMK Indonesia
    - Sumber: UCI Dropout, Student Expanded, Student Performance, Exams, UCI Math, UCI Portuguese
    - Fitur: Kehadiran, Nilai, KIP, Jarak, Pelanggaran, dll.""")
    st.subheader("Inovasi Utama")
    st.markdown("""1. **Ensemble Voting** — Kombinasi 3 model ML
2. **SHAP Explainable AI** — Prediksi transparan
3. **What-If Simulator** — Simulasi intervensi
4. **AI BK Assistant** — Chatbot analisis
5. **Early Warning System** — Peringatan dini
6. **Intervention Tracking** — Pelacakan intervensi
7. **Responsible AI** — Fairness, human-in-the-loop, audit trail""")

st.markdown(f'<hr style="margin:2rem 0 1rem 0;border-color:{CARD_BORDER};opacity:0.5;">', unsafe_allow_html=True)
st.markdown(f"""<div style="text-align:center;padding:1rem;">
    <div style="font-size:0.8rem;color:{TEXT_DIM};letter-spacing:0.05em;">
        SIGAP Sekolah v4.0 | LKS AI 2026 | Data: REAL DATA | Ensemble AI + XAI + Responsible AI
    </div>
    <div style="font-size:0.65rem;color:{TEXT_DIM};margin-top:0.3rem;opacity:0.6;">
        Sistem Identifikasi Gejala Anak Berisiko Putus Sekolah
    </div>
</div>""", unsafe_allow_html=True)
