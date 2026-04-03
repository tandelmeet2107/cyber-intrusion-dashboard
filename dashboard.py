"""
╔═══════════════════════════════════════════════════════════════════╗
║     CYBER INTRUSION LOG ANALYZER — Streamlit Dashboard           ║
║     NSL-KDD Dataset  |  Final Year Project                       ║
╚═══════════════════════════════════════════════════════════════════╝

Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber Intrusion Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark Cyber Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── root colours ── */
:root {
    --bg-dark:    #0a0e1a;
    --bg-card:    #111827;
    --bg-card2:   #1a2235;
    --accent:     #00d4ff;
    --accent2:    #7c3aed;
    --danger:     #ef4444;
    --warn:       #f59e0b;
    --ok:         #10b981;
    --text:       #e2e8f0;
    --text-muted: #94a3b8;
}

/* ── global ── */
.stApp { background-color: var(--bg-dark); color: var(--text); }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* metric cards */
[data-testid="metric-container"] {
    background: var(--bg-card2);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 0 18px rgba(0,212,255,.08);
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
[data-testid="stMetricDelta"] { font-size: .85rem !important; }

/* section header */
.section-header {
    color: var(--accent);
    font-size: 1.35rem;
    font-weight: 700;
    border-bottom: 2px solid #1e3a5f;
    padding-bottom: 6px;
    margin-bottom: 18px;
    letter-spacing: .6px;
}

/* info / alert cards */
.info-card {
    background: var(--bg-card);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: .9rem;
}
.danger-card {
    background: #1f0d0d;
    border-left: 4px solid var(--danger);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}
.warn-card {
    background: #1a1200;
    border-left: 4px solid var(--warn);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}
.ok-card {
    background: #0a1f15;
    border-left: 4px solid var(--ok);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}

/* plotly charts — transparent bg */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* stTabs */
[data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 10px; }
[data-baseweb="tab"] { color: var(--text-muted) !important; }
[aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

/* dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }

/* selectbox / upload */
[data-testid="stFileUploader"] {
    border: 2px dashed #1e3a5f;
    border-radius: 10px;
    background: var(--bg-card);
    padding: 10px;
}

/* title block */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #2563eb33;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,212,255,.12);
}
.hero h1 { font-size: 2.2rem; color: var(--accent); margin: 0; letter-spacing: 1px; }
.hero p  { color: var(--text-muted); margin: 6px 0 0; font-size: 1rem; }

/* badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 600;
}
.badge-danger  { background:#3f1010; color:#f87171; border:1px solid #7f1d1d; }
.badge-warn    { background:#2d1f00; color:#fbbf24; border:1px solid #78350f; }
.badge-ok      { background:#052e16; color:#34d399; border:1px solid #064e3b; }
.badge-neutral { background:#1e293b; color:#94a3b8; border:1px solid #334155; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY CHART THEME
# ─────────────────────────────────────────────
CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
)
COLOR_SEQ   = px.colors.qualitative.Dark24
COLOR_ATTACK = {"normal": "#10b981", "DoS": "#ef4444", "Probe": "#f59e0b",
                 "R2L": "#8b5cf6", "U2R": "#ec4899"}
COLOR_SEV   = {"none": "#10b981", "medium": "#f59e0b",
               "high": "#f97316", "critical": "#ef4444"}
SEVERITY_ORDER = ["none", "medium", "high", "critical"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
REQUIRED_COLS = {
    "timestamp", "protocol_type", "service", "flag", "is_attack",
    "label", "attack_category", "severity", "risk_score",
    "src_bytes", "dst_bytes", "hour", "day_of_week",
}

DAY_MAP = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


@st.cache_data(show_spinner=False)
def load_default_data(path: str) -> pd.DataFrame:
    df = pd.read_csv("nslkdd_powerbi.csv")
    return preprocess(df)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].notna().any():
            df["date"] = df["timestamp"].dt.date
            if "hour" not in df.columns:
                df["hour"] = df["timestamp"].dt.hour
            if "day_of_week" not in df.columns:
                df["day_of_week"] = df["timestamp"].dt.dayofweek

    # derived columns if missing
    if "attack_category" not in df.columns and "label" in df.columns:
        mapping = {
            "neptune": "DoS", "smurf": "DoS", "back": "DoS", "land": "DoS",
            "pod": "DoS", "teardrop": "DoS", "portsweep": "Probe",
            "ipsweep": "Probe", "nmap": "Probe", "satan": "Probe",
            "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
            "multihop": "R2L", "phf": "R2L", "spy": "R2L",
            "warezclient": "R2L", "warezmaster": "R2L",
            "buffer_overflow": "U2R", "loadmodule": "U2R",
            "perl": "U2R", "rootkit": "U2R", "normal": "normal",
        }
        df["attack_category"] = df["label"].map(mapping).fillna("Other")

    if "is_attack" not in df.columns and "attack_category" in df.columns:
        df["is_attack"] = (df["attack_category"] != "normal").astype(int)

    if "severity" not in df.columns:
        def _sev(row):
            if row.get("attack_category") == "normal":
                return "none"
            cat = row.get("attack_category", "")
            if cat == "U2R":
                return "critical"
            if cat == "R2L":
                return "high"
            if cat == "DoS":
                return "high"
            return "medium"
        df["severity"] = df.apply(_sev, axis=1)

    if "risk_score" not in df.columns:
        sev_score = {"none": 0, "medium": 50, "high": 150, "critical": 250}
        df["risk_score"] = df["severity"].map(sev_score).fillna(0)

    if "day_of_week" in df.columns:
        df["day_name"] = df["day_of_week"].map(DAY_MAP)

    return df


def validate_columns(df: pd.DataFrame) -> tuple[bool, list]:
    missing = REQUIRED_COLS - set(df.columns)
    # allow a partial match — show warning but proceed
    return len(missing) == 0, list(missing)


def load_uploaded_file(uploaded) -> pd.DataFrame | None:
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)
        elif name.endswith(".json"):
            df = pd.read_json(uploaded)
        else:
            st.error("Unsupported file type. Please upload CSV, Excel, Parquet, or JSON.")
            return None
        return preprocess(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def make_chart(fig, height=380):
    fig.update_layout(height=height, **CHART_THEME,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0'>
        <div style='font-size:2.4rem'>🛡️</div>
        <div style='color:#00d4ff; font-weight:700; font-size:1.1rem'>CYBER INTRUSION</div>
        <div style='color:#94a3b8; font-size:.8rem'>LOG ANALYZER</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Dataset chooser ──
    st.markdown("### 📂 Dataset")
    data_source = st.radio("Source", ["Default Dataset (NSL-KDD)", "Upload Custom Dataset"],
                           label_visibility="collapsed")

    df_raw = None
    dataset_name = "NSL-KDD"

    if data_source == "Default Dataset (NSL-KDD)":
        default_path = os.path.join(os.path.dirname(__file__), "nslkdd_powerbi.csv")
        if os.path.exists(default_path):
            with st.spinner("Loading dataset…"):
                df_raw = load_default_data(default_path)
            st.success(f"✅ {len(df_raw):,} records loaded")
        else:
            st.error("Default CSV not found. Place `nslkdd_powerbi.csv` beside this script.")
    else:
        uploaded = st.file_uploader(
            "Upload your log file",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            help="CSV, Excel, Parquet or JSON supported. Must contain network/intrusion log columns.",
        )
        if uploaded:
            with st.spinner("Processing…"):
                df_raw = load_uploaded_file(uploaded)
            if df_raw is not None:
                dataset_name = uploaded.name
                ok, missing = validate_columns(df_raw)
                if not ok:
                    st.warning(f"⚠️ Some expected columns missing: `{'`, `'.join(missing)}`  \n"
                               "Dashboard will render available features only.")
                st.success(f"✅ {len(df_raw):,} records loaded")
        else:
            st.info("☝️ Upload a file to begin")

    # ── Filters (only when data is loaded) ──
    if df_raw is not None:
        st.divider()
        st.markdown("### 🔽 Filters")

        if "attack_category" in df_raw.columns:
            cats = sorted(df_raw["attack_category"].dropna().unique())
            sel_cats = st.multiselect("Attack Category", cats, default=cats, key="filter_cat")
        else:
            sel_cats = None

        if "severity" in df_raw.columns:
            sevs = [s for s in SEVERITY_ORDER if s in df_raw["severity"].unique()]
            sel_sevs = st.multiselect("Severity", sevs, default=sevs, key="filter_sev")
        else:
            sel_sevs = None

        if "protocol_type" in df_raw.columns:
            protos = sorted(df_raw["protocol_type"].dropna().unique())
            sel_proto = st.multiselect("Protocol", protos, default=protos, key="filter_proto")
        else:
            sel_proto = None

    # ── Navigation ──
    if df_raw is not None:
        st.divider()
        st.markdown("### 📑 Pages")
        page = st.radio(
            "Navigate",
            ["🏠 Overview", "⚔️ Attack Analysis", "🌐 Network Traffic",
             "⏰ Time Analysis", "📡 Anomaly & Risk", "🔍 Raw Data"],
            label_visibility="collapsed",
        )
    else:
        page = None

    st.divider()
    st.markdown("<div style='color:#475569; font-size:.72rem; text-align:center'>"
                "Final Year Project · NSL-KDD Cyber Intrusion Analyzer<br>"
                "Built with Streamlit & Plotly</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛡️ Cyber Intrusion Log Analyzer</h1>
  <p>AI-Powered Network Security Dashboard &nbsp;|&nbsp; NSL-KDD Dataset &nbsp;|&nbsp; Final Year Project</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# NO DATA STATE
# ─────────────────────────────────────────────
if df_raw is None:
    st.markdown("""
    <div class="info-card" style='text-align:center; padding:40px'>
        <div style='font-size:3rem'>📂</div>
        <h3 style='color:#00d4ff'>No Dataset Loaded</h3>
        <p style='color:#94a3b8'>Select <b>Default Dataset</b> or upload your own file from the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_raw.copy()
if sel_cats and "attack_category" in df.columns:
    df = df[df["attack_category"].isin(sel_cats)]
if sel_sevs and "severity" in df.columns:
    df = df[df["severity"].isin(sel_sevs)]
if sel_proto and "protocol_type" in df.columns:
    df = df[df["protocol_type"].isin(sel_proto)]

if len(df) == 0:
    st.warning("⚠️ No records match your current filters. Please adjust the sidebar filters.")
    st.stop()


# ─────────────────────────────────────────────
# SHARED METRICS
# ─────────────────────────────────────────────
total    = len(df)
attacks  = int(df["is_attack"].sum()) if "is_attack" in df.columns else 0
normals  = total - attacks
atk_pct  = attacks / total * 100 if total else 0
avg_risk = df["risk_score"].mean() if "risk_score" in df.columns else 0
crit_cnt = (df["severity"] == "critical").sum() if "severity" in df.columns else 0
anomalies = int(df["iso_anomaly"].sum()) if "iso_anomaly" in df.columns else 0


# ══════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════
if page == "🏠 Overview":
    # ── KPI Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events",    f"{total:,}")
    c2.metric("Attacks Detected", f"{attacks:,}",  delta=f"{atk_pct:.1f}% of traffic",
              delta_color="inverse")
    c3.metric("Normal Traffic",  f"{normals:,}")
    c4.metric("Avg Risk Score",  f"{avg_risk:.0f}", delta="out of 300",
              delta_color="off")
    c5.metric("Critical Events", f"{crit_cnt:,}",
              delta="🔴 Needs attention" if crit_cnt > 0 else "✅ None",
              delta_color="inverse" if crit_cnt > 0 else "normal")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 ──
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header">Traffic Composition</div>', unsafe_allow_html=True)
        pie_data = pd.DataFrame({
            "Type": ["Normal", "Attack"],
            "Count": [normals, attacks],
        })
        fig = px.pie(pie_data, values="Count", names="Type",
                     color="Type",
                     color_discrete_map={"Normal": "#10b981", "Attack": "#ef4444"},
                     hole=0.55)
        fig.update_traces(textinfo="percent+label", textfont_size=13,
                          pull=[0, 0.06])
        fig.update_layout(showlegend=True, **CHART_THEME,
                          height=340, margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Attack Category Breakdown</div>', unsafe_allow_html=True)
        if "attack_category" in df.columns:
            cat_counts = df["attack_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig2 = px.bar(cat_counts, x="Category", y="Count",
                          color="Category",
                          color_discrete_map=COLOR_ATTACK,
                          text="Count")
            fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig2.update_layout(**CHART_THEME, height=340,
                               margin=dict(l=10, r=10, t=20, b=10),
                               showlegend=False,
                               xaxis_title="", yaxis_title="Events")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2 ──
    col_c, col_d, col_e = st.columns([1.2, 1, 0.8])

    with col_c:
        st.markdown('<div class="section-header">Severity Distribution</div>', unsafe_allow_html=True)
        if "severity" in df.columns:
            sev_data = df["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0).reset_index()
            sev_data.columns = ["Severity", "Count"]
            colors = [COLOR_SEV.get(s, "#94a3b8") for s in sev_data["Severity"]]
            fig3 = px.bar(sev_data, x="Severity", y="Count",
                          color="Severity",
                          color_discrete_map=COLOR_SEV,
                          category_orders={"Severity": SEVERITY_ORDER},
                          text="Count")
            fig3.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig3.update_layout(**CHART_THEME, height=300,
                               margin=dict(l=10, r=10, t=20, b=10),
                               showlegend=False, xaxis_title="", yaxis_title="Events")
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Protocol Mix</div>', unsafe_allow_html=True)
        if "protocol_type" in df.columns:
            proto_data = df["protocol_type"].value_counts().reset_index()
            proto_data.columns = ["Protocol", "Count"]
            fig4 = px.pie(proto_data, values="Count", names="Protocol",
                          color_discrete_sequence=["#00d4ff", "#7c3aed", "#f59e0b"],
                          hole=0.4)
            fig4.update_traces(textinfo="percent+label", textfont_size=12)
            fig4.update_layout(**CHART_THEME, height=300,
                               margin=dict(l=10, r=10, t=20, b=10),
                               showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    with col_e:
        st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
        if "iso_anomaly" in df.columns:
            iso_pct = df["iso_anomaly"].mean() * 100
            st.markdown(f"""
            <div class="info-card">🧠 <b>Isolation Forest</b><br>
              Anomaly Rate: <span style='color:#00d4ff; font-weight:700'>{iso_pct:.1f}%</span><br>
              Anomalies: <b>{anomalies:,}</b>
            </div>""", unsafe_allow_html=True)
        if "km_cluster" in df.columns:
            n_clusters = df["km_cluster"].nunique()
            st.markdown(f"""
            <div class="info-card">📊 <b>KMeans Clusters</b><br>
              Found <span style='color:#00d4ff; font-weight:700'>{n_clusters}</span> traffic clusters
            </div>""", unsafe_allow_html=True)
        if "is_night" in df.columns:
            night_atk = df[df["is_night"] == 1]["is_attack"].mean() * 100 if "is_attack" in df.columns else 0
            day_atk   = df[df["is_night"] == 0]["is_attack"].mean() * 100 if "is_attack" in df.columns else 0
            st.markdown(f"""
            <div class="warn-card">🌙 <b>Night vs Day</b><br>
              Night attack rate: <b>{night_atk:.1f}%</b><br>
              Day attack rate: <b>{day_atk:.1f}%</b>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: ATTACK ANALYSIS
# ══════════════════════════════════════════════
elif page == "⚔️ Attack Analysis":
    st.markdown('<div class="section-header">⚔️ Attack Analysis</div>', unsafe_allow_html=True)

    # ── Top attack types ──
    col1, col2 = st.columns(2)
    with col1:
        if "label" in df.columns:
            lbl_counts = df[df["is_attack"] == 1]["label"].value_counts().head(12).reset_index()
            lbl_counts.columns = ["Attack Type", "Count"]
            fig = px.bar(lbl_counts, x="Count", y="Attack Type", orientation="h",
                         color="Count", color_continuous_scale="Reds",
                         title="Top 12 Attack Types")
            fig.update_layout(**CHART_THEME, height=400,
                              margin=dict(l=10, r=10, t=40, b=10),
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "attack_category" in df.columns and "severity" in df.columns:
            heat_data = df[df["is_attack"] == 1].groupby(
                ["attack_category", "severity"]).size().reset_index(name="Count")
            heat_pivot = heat_data.pivot(index="attack_category",
                                         columns="severity", values="Count").fillna(0)
            # reorder columns
            sev_cols = [c for c in SEVERITY_ORDER if c in heat_pivot.columns]
            heat_pivot = heat_pivot[sev_cols]
            fig2 = px.imshow(heat_pivot,
                             color_continuous_scale="RdYlGn_r",
                             title="Attack Category × Severity Heatmap",
                             labels=dict(x="Severity", y="Attack Category", color="Count"),
                             text_auto=True)
            fig2.update_layout(**CHART_THEME, height=400,
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Service vs Attack ──
    col3, col4 = st.columns(2)
    with col3:
        if "service" in df.columns and "is_attack" in df.columns:
            svc_atk = df.groupby("service")["is_attack"].agg(
                ["sum", "count"]).reset_index()
            svc_atk.columns = ["Service", "Attacks", "Total"]
            svc_atk["Attack Rate"] = svc_atk["Attacks"] / svc_atk["Total"] * 100
            svc_atk = svc_atk.sort_values("Attacks", ascending=False).head(12)
            fig3 = px.scatter(svc_atk, x="Total", y="Attack Rate",
                              size="Attacks", color="Attack Rate",
                              text="Service",
                              color_continuous_scale="YlOrRd",
                              title="Service: Volume vs Attack Rate",
                              labels={"Total": "Total Events", "Attack Rate": "Attack Rate (%)"})
            fig3.update_traces(textposition="top center", textfont_size=9)
            fig3.update_layout(**CHART_THEME, height=380,
                               margin=dict(l=10, r=10, t=40, b=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if "flag" in df.columns and "attack_category" in df.columns:
            flag_cat = df.groupby(["flag", "attack_category"]).size().reset_index(name="Count")
            fig4 = px.bar(flag_cat, x="flag", y="Count",
                          color="attack_category",
                          color_discrete_map=COLOR_ATTACK,
                          title="Connection Flag by Attack Category",
                          barmode="stack")
            fig4.update_layout(**CHART_THEME, height=380,
                               margin=dict(l=10, r=10, t=40, b=10),
                               xaxis_title="Connection Flag",
                               legend_title="Category")
            st.plotly_chart(fig4, use_container_width=True)

    # ── Risk score distribution ──
    st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
    if "risk_score" in df.columns and "attack_category" in df.columns:
        fig5 = px.violin(df[df["is_attack"] == 1], x="attack_category", y="risk_score",
                         color="attack_category",
                         color_discrete_map=COLOR_ATTACK,
                         box=True, points="outliers",
                         title="Risk Score by Attack Category")
        fig5.update_layout(**CHART_THEME, height=380,
                           margin=dict(l=10, r=10, t=40, b=10),
                           showlegend=False, xaxis_title="", yaxis_title="Risk Score")
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE: NETWORK TRAFFIC
# ══════════════════════════════════════════════
elif page == "🌐 Network Traffic":
    st.markdown('<div class="section-header">🌐 Network Traffic Analysis</div>', unsafe_allow_html=True)

    # ── Bytes analysis ──
    col1, col2 = st.columns(2)
    with col1:
        if "src_bytes" in df.columns and "attack_category" in df.columns:
            df_clip = df[df["src_bytes"] < df["src_bytes"].quantile(0.97)].copy()
            fig = px.box(df_clip, x="attack_category", y="src_bytes",
                         color="attack_category",
                         color_discrete_map=COLOR_ATTACK,
                         title="Source Bytes by Attack Category")
            fig.update_layout(**CHART_THEME, height=360,
                              margin=dict(l=10, r=10, t=40, b=10),
                              showlegend=False, xaxis_title="", yaxis_title="Source Bytes")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "dst_bytes" in df.columns and "attack_category" in df.columns:
            df_clip2 = df[df["dst_bytes"] < df["dst_bytes"].quantile(0.97)].copy()
            fig2 = px.box(df_clip2, x="attack_category", y="dst_bytes",
                          color="attack_category",
                          color_discrete_map=COLOR_ATTACK,
                          title="Destination Bytes by Attack Category")
            fig2.update_layout(**CHART_THEME, height=360,
                               margin=dict(l=10, r=10, t=40, b=10),
                               showlegend=False, xaxis_title="", yaxis_title="Destination Bytes")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Protocol + Service ──
    col3, col4 = st.columns(2)
    with col3:
        if "protocol_type" in df.columns and "is_attack" in df.columns:
            proto_grp = df.groupby(["protocol_type", "is_attack"]).size().reset_index(name="Count")
            proto_grp["Traffic"] = proto_grp["is_attack"].map({0: "Normal", 1: "Attack"})
            fig3 = px.bar(proto_grp, x="protocol_type", y="Count",
                          color="Traffic",
                          color_discrete_map={"Normal": "#10b981", "Attack": "#ef4444"},
                          barmode="group",
                          title="Protocol: Normal vs Attack Volume")
            fig3.update_layout(**CHART_THEME, height=340,
                               margin=dict(l=10, r=10, t=40, b=10),
                               xaxis_title="Protocol", yaxis_title="Events")
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if "service" in df.columns:
            top_svc = df["service"].value_counts().head(10).reset_index()
            top_svc.columns = ["Service", "Count"]
            fig4 = px.pie(top_svc, values="Count", names="Service",
                          title="Top 10 Services (Traffic Share)",
                          color_discrete_sequence=COLOR_SEQ)
            fig4.update_traces(textinfo="label+percent")
            fig4.update_layout(**CHART_THEME, height=340,
                               margin=dict(l=10, r=10, t=40, b=10),
                               showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Connection features ──
    col5, col6 = st.columns(2)
    with col5:
        if "serror_rate" in df.columns and "rerror_rate" in df.columns and "attack_category" in df.columns:
            sample_df = df.sample(min(3000, len(df)), random_state=42)
            fig5 = px.scatter(sample_df, x="serror_rate", y="rerror_rate",
                              color="attack_category",
                              color_discrete_map=COLOR_ATTACK,
                              opacity=0.6, size_max=6,
                              title="SYN Error Rate vs REJ Error Rate",
                              labels={"serror_rate": "SYN Error Rate",
                                      "rerror_rate": "REJ Error Rate"})
            fig5.update_layout(**CHART_THEME, height=340,
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig5, use_container_width=True)

    with col6:
        if "count" in df.columns and "same_srv_rate" in df.columns and "attack_category" in df.columns:
            sample_df2 = df.sample(min(3000, len(df)), random_state=42)
            fig6 = px.scatter(sample_df2, x="count", y="same_srv_rate",
                              color="attack_category",
                              color_discrete_map=COLOR_ATTACK,
                              opacity=0.55, size_max=6,
                              title="Connection Count vs Same-Service Rate",
                              labels={"count": "Connection Count",
                                      "same_srv_rate": "Same Service Rate"})
            fig6.update_layout(**CHART_THEME, height=340,
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig6, use_container_width=True)

    # ── Top IPs ──
    if "src_ip" in df.columns:
        st.markdown('<div class="section-header">🔌 Top Source IPs by Activity</div>',
                    unsafe_allow_html=True)
        ip_tab1, ip_tab2 = st.tabs(["Most Active IPs", "Most Malicious IPs"])
        with ip_tab1:
            top_ips = df["src_ip"].value_counts().head(15).reset_index()
            top_ips.columns = ["IP", "Events"]
            fig_ip = px.bar(top_ips, x="Events", y="IP", orientation="h",
                            color="Events", color_continuous_scale="Blues",
                            title="Top 15 Source IPs by Event Count")
            fig_ip.update_layout(**CHART_THEME, height=420,
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 yaxis=dict(autorange="reversed"),
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_ip, use_container_width=True)

        with ip_tab2:
            if "is_attack" in df.columns:
                malicious_ips = (df[df["is_attack"] == 1]["src_ip"]
                                 .value_counts().head(15).reset_index())
                malicious_ips.columns = ["IP", "Attacks"]
                fig_mal = px.bar(malicious_ips, x="Attacks", y="IP", orientation="h",
                                 color="Attacks", color_continuous_scale="Reds",
                                 title="Top 15 Source IPs by Attack Count")
                fig_mal.update_layout(**CHART_THEME, height=420,
                                      margin=dict(l=10, r=10, t=40, b=10),
                                      yaxis=dict(autorange="reversed"),
                                      coloraxis_showscale=False)
                st.plotly_chart(fig_mal, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE: TIME ANALYSIS
# ══════════════════════════════════════════════
elif page == "⏰ Time Analysis":
    st.markdown('<div class="section-header">⏰ Temporal Attack Patterns</div>',
                unsafe_allow_html=True)

    # ── Hourly ──
    if "hour" in df.columns and "is_attack" in df.columns:
        hourly = df.groupby(["hour", "attack_category"]).size().reset_index(name="Count") \
            if "attack_category" in df.columns else \
            df.groupby("hour")["is_attack"].sum().reset_index(name="Attacks")

        if "attack_category" in hourly.columns:
            fig_h = px.bar(hourly, x="hour", y="Count",
                           color="attack_category",
                           color_discrete_map=COLOR_ATTACK,
                           title="Hourly Traffic — Attack Category",
                           labels={"hour": "Hour of Day", "Count": "Events"})
        else:
            fig_h = px.bar(hourly, x="hour", y="Attacks",
                           title="Hourly Attack Count",
                           color_discrete_sequence=["#ef4444"])
        fig_h.update_layout(**CHART_THEME, height=360,
                            margin=dict(l=10, r=10, t=40, b=10),
                            bargap=0.1)
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Daily ──
    col_a, col_b = st.columns(2)
    with col_a:
        if "day_of_week" in df.columns and "is_attack" in df.columns:
            daily = df.groupby("day_of_week").agg(
                Total=("is_attack", "count"),
                Attacks=("is_attack", "sum"),
            ).reset_index()
            daily["Normal"] = daily["Total"] - daily["Attacks"]
            daily["Day"] = daily["day_of_week"].map(DAY_MAP)
            day_melt = daily.melt(id_vars="Day", value_vars=["Normal", "Attacks"])
            day_melt.columns = ["Day", "Type", "Count"]
            fig_d = px.bar(day_melt, x="Day", y="Count",
                           color="Type",
                           color_discrete_map={"Normal": "#10b981", "Attacks": "#ef4444"},
                           barmode="stack",
                           title="Daily Traffic Distribution",
                           category_orders={"Day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]})
            fig_d.update_layout(**CHART_THEME, height=340,
                                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_d, use_container_width=True)

    with col_b:
        if "hour" in df.columns and "day_of_week" in df.columns and "is_attack" in df.columns:
            heat = df.groupby(["day_of_week", "hour"])["is_attack"].sum().reset_index()
            heat["Day"] = heat["day_of_week"].map(DAY_MAP)
            heat_piv = heat.pivot(index="Day", columns="hour", values="is_attack").fillna(0)
            day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            heat_piv = heat_piv.reindex([d for d in day_order if d in heat_piv.index])
            fig_hm = px.imshow(heat_piv,
                               color_continuous_scale="YlOrRd",
                               title="Attack Heatmap: Day × Hour",
                               labels=dict(x="Hour", y="Day", color="Attacks"))
            fig_hm.update_layout(**CHART_THEME, height=340,
                                 margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_hm, use_container_width=True)

    # ── Night vs Day ──
    if "is_night" in df.columns:
        st.markdown('<div class="section-header">🌙 Night vs. Day Attack Comparison</div>',
                    unsafe_allow_html=True)
        col_c, col_d = st.columns(2)
        with col_c:
            night_grp = df.groupby(["is_night", "attack_category"]).size().reset_index(name="Count")
            night_grp["Period"] = night_grp["is_night"].map({0: "☀️ Day", 1: "🌙 Night"})
            fig_n = px.bar(night_grp, x="Period", y="Count",
                           color="attack_category",
                           color_discrete_map=COLOR_ATTACK,
                           barmode="stack",
                           title="Attack Category: Day vs Night")
            fig_n.update_layout(**CHART_THEME, height=320,
                                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_n, use_container_width=True)
        with col_d:
            night_rate = df.groupby("is_night")["is_attack"].mean().reset_index()
            night_rate["Period"] = night_rate["is_night"].map({0: "☀️ Day", 1: "🌙 Night"})
            night_rate["Attack Rate %"] = night_rate["is_attack"] * 100
            fig_r = px.bar(night_rate, x="Period", y="Attack Rate %",
                           color="Period",
                           color_discrete_sequence=["#f59e0b", "#7c3aed"],
                           text=night_rate["Attack Rate %"].map("{:.1f}%".format),
                           title="Attack Rate: Day vs Night")
            fig_r.update_traces(textposition="outside")
            fig_r.update_layout(**CHART_THEME, height=320,
                                margin=dict(l=10, r=10, t=40, b=10),
                                showlegend=False, yaxis_title="Attack Rate (%)")
            st.plotly_chart(fig_r, use_container_width=True)

    # ── Time-series trend ──
    if "date" in df.columns and "is_attack" in df.columns:
        st.markdown('<div class="section-header">📈 Daily Trend</div>', unsafe_allow_html=True)
        trend = df.groupby("date").agg(
            Total=("is_attack", "count"),
            Attacks=("is_attack", "sum"),
        ).reset_index()
        trend["Attack Rate"] = trend["Attacks"] / trend["Total"] * 100
        fig_t = make_subplots(specs=[[{"secondary_y": True}]])
        fig_t.add_trace(go.Bar(x=trend["date"], y=trend["Total"],
                               name="Total Events", marker_color="#1e3a5f",
                               opacity=0.8), secondary_y=False)
        fig_t.add_trace(go.Scatter(x=trend["date"], y=trend["Attacks"],
                                   name="Attacks", mode="lines+markers",
                                   line=dict(color="#ef4444", width=2),
                                   marker=dict(size=4)), secondary_y=False)
        fig_t.add_trace(go.Scatter(x=trend["date"], y=trend["Attack Rate"],
                                   name="Attack Rate %", mode="lines",
                                   line=dict(color="#f59e0b", width=2, dash="dot")),
                        secondary_y=True)
        fig_t.update_layout(**CHART_THEME, height=360,
                            margin=dict(l=10, r=10, t=40, b=10),
                            title="Daily Events, Attacks & Attack Rate",
                            legend=dict(orientation="h", y=1.1))
        fig_t.update_yaxes(title_text="Events", secondary_y=False)
        fig_t.update_yaxes(title_text="Attack Rate (%)", secondary_y=True)
        st.plotly_chart(fig_t, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE: ANOMALY & RISK
# ══════════════════════════════════════════════
elif page == "📡 Anomaly & Risk":
    st.markdown('<div class="section-header">📡 Anomaly Detection & Risk Intelligence</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── Isolation Forest ──
    with col1:
        if "iso_anomaly" in df.columns and "attack_category" in df.columns:
            iso_grp = df.groupby(["attack_category", "iso_anomaly"]).size().reset_index(name="Count")
            iso_grp["Prediction"] = iso_grp["iso_anomaly"].map({0: "Normal", 1: "Anomaly"})
            fig_iso = px.bar(iso_grp, x="attack_category", y="Count",
                             color="Prediction",
                             color_discrete_map={"Normal": "#10b981", "Anomaly": "#ef4444"},
                             barmode="group",
                             title="Isolation Forest: Anomalies by Attack Category")
            fig_iso.update_layout(**CHART_THEME, height=360,
                                  margin=dict(l=10, r=10, t=40, b=10),
                                  xaxis_title="", yaxis_title="Events")
            st.plotly_chart(fig_iso, use_container_width=True)

    with col2:
        if "iso_anomaly_score" in df.columns and "attack_category" in df.columns:
            fig_score = px.violin(df, x="attack_category", y="iso_anomaly_score",
                                  color="attack_category",
                                  color_discrete_map=COLOR_ATTACK,
                                  box=True, points=False,
                                  title="Isolation Forest Score Distribution")
            fig_score.update_layout(**CHART_THEME, height=360,
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    showlegend=False, xaxis_title="",
                                    yaxis_title="Anomaly Score (lower = more anomalous)")
            st.plotly_chart(fig_score, use_container_width=True)

    # ── KMeans ──
    col3, col4 = st.columns(2)
    with col3:
        if "km_cluster" in df.columns and "attack_category" in df.columns:
            km_grp = df.groupby(["km_cluster", "attack_category"]).size().reset_index(name="Count")
            fig_km = px.bar(km_grp, x="km_cluster", y="Count",
                            color="attack_category",
                            color_discrete_map=COLOR_ATTACK,
                            barmode="stack",
                            title="KMeans Cluster Composition",
                            labels={"km_cluster": "Cluster ID"})
            fig_km.update_layout(**CHART_THEME, height=360,
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 xaxis_title="Cluster ID", yaxis_title="Events")
            st.plotly_chart(fig_km, use_container_width=True)

    with col4:
        if "km_cluster" in df.columns and "risk_score" in df.columns:
            cluster_risk = df.groupby("km_cluster")["risk_score"].mean().reset_index()
            cluster_risk.columns = ["Cluster", "Avg Risk Score"]
            cluster_risk["Cluster"] = "Cluster " + cluster_risk["Cluster"].astype(str)
            fig_cr = px.bar(cluster_risk, x="Cluster", y="Avg Risk Score",
                            color="Avg Risk Score",
                            color_continuous_scale="RdYlGn_r",
                            title="Average Risk Score per KMeans Cluster",
                            text=cluster_risk["Avg Risk Score"].round(1))
            fig_cr.update_traces(textposition="outside")
            fig_cr.update_layout(**CHART_THEME, height=360,
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_cr, use_container_width=True)

    # ── Risk Score Histogram ──
    st.markdown('<div class="section-header">📊 Risk Score Distribution</div>',
                unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        if "risk_score" in df.columns:
            fig_rh = px.histogram(df, x="risk_score", nbins=40,
                                  color_discrete_sequence=["#7c3aed"],
                                  title="Overall Risk Score Distribution")
            fig_rh.add_vline(x=avg_risk, line_dash="dash", line_color="#f59e0b",
                             annotation_text=f"Mean: {avg_risk:.0f}",
                             annotation_position="top right")
            fig_rh.update_layout(**CHART_THEME, height=320,
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 xaxis_title="Risk Score", yaxis_title="Count")
            st.plotly_chart(fig_rh, use_container_width=True)

    with col6:
        if "risk_score" in df.columns and "severity" in df.columns:
            fig_rb = px.box(df, x="severity", y="risk_score",
                            color="severity",
                            color_discrete_map=COLOR_SEV,
                            category_orders={"severity": SEVERITY_ORDER},
                            title="Risk Score by Severity Level")
            fig_rb.update_layout(**CHART_THEME, height=320,
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 showlegend=False, xaxis_title="Severity")
            st.plotly_chart(fig_rb, use_container_width=True)

    # ── Confusion matrix: iso_anomaly vs is_attack ──
    if "iso_anomaly" in df.columns and "is_attack" in df.columns:
        st.markdown('<div class="section-header">🧪 Anomaly Detector Accuracy</div>',
                    unsafe_allow_html=True)
        from sklearn.metrics import confusion_matrix  # type: ignore
        try:
            cm = confusion_matrix(df["is_attack"], df["iso_anomaly"])
            tn, fp, fn, tp = cm.ravel()
            acc = (tp + tn) / (tp + tn + fp + fn) * 100
            prec = tp / (tp + fp) * 100 if (tp + fp) else 0
            rec  = tp / (tp + fn) * 100 if (tp + fn) else 0

            cm_df = pd.DataFrame(cm,
                                 index=["Actual Normal", "Actual Attack"],
                                 columns=["Predicted Normal", "Predicted Attack"])
            fig_cm = px.imshow(cm_df, text_auto=True,
                               color_continuous_scale="Blues",
                               title="Isolation Forest — Confusion Matrix")
            fig_cm.update_layout(**CHART_THEME, height=320,
                                 margin=dict(l=10, r=10, t=40, b=10))

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Accuracy",  f"{acc:.1f}%")
            mc2.metric("Precision", f"{prec:.1f}%")
            mc3.metric("Recall",    f"{rec:.1f}%")
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception:
            st.info("Could not compute confusion matrix for this dataset.")


# ══════════════════════════════════════════════
#  PAGE: RAW DATA
# ══════════════════════════════════════════════
elif page == "🔍 Raw Data":
    st.markdown('<div class="section-header">🔍 Raw Log Explorer</div>', unsafe_allow_html=True)

    # Search & filter
    search_col, cat_col, sev_col = st.columns([2, 1, 1])
    with search_col:
        search = st.text_input("🔎 Search (IP, label, service…)", placeholder="e.g.  neptune  or  192.168")
    with cat_col:
        if "attack_category" in df.columns:
            cats_raw = ["All"] + sorted(df["attack_category"].dropna().unique())
            sel_raw_cat = st.selectbox("Category", cats_raw)
        else:
            sel_raw_cat = "All"
    with sev_col:
        if "severity" in df.columns:
            sevs_raw = ["All"] + SEVERITY_ORDER
            sel_raw_sev = st.selectbox("Severity", sevs_raw)
        else:
            sel_raw_sev = "All"

    display_df = df.copy()
    if search:
        mask = np.zeros(len(display_df), dtype=bool)
        for col in display_df.select_dtypes(include="object").columns:
            mask |= display_df[col].astype(str).str.contains(search, case=False, na=False)
        display_df = display_df[mask]
    if sel_raw_cat != "All" and "attack_category" in display_df.columns:
        display_df = display_df[display_df["attack_category"] == sel_raw_cat]
    if sel_raw_sev != "All" and "severity" in display_df.columns:
        display_df = display_df[display_df["severity"] == sel_raw_sev]

    st.caption(f"Showing **{len(display_df):,}** of **{len(df):,}** records")
    st.dataframe(
        display_df.head(500).reset_index(drop=True),
        use_container_width=True,
        height=450,
    )

    st.divider()
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Filtered CSV", csv_bytes,
                           file_name="filtered_logs.csv", mime="text/csv")
    with dl_col2:
        st.info(f"📦 Total filtered rows: **{len(display_df):,}**  |  "
                f"Columns: **{len(display_df.columns)}**")

    # ── Column stats ──
    st.markdown('<div class="section-header">📊 Column Statistics</div>', unsafe_allow_html=True)
    num_cols = display_df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        st.dataframe(display_df[num_cols].describe().T.style.format("{:.2f}"),
                     use_container_width=True)
