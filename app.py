import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import shap
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="AHASS Prakiraan Permintaan",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    *, html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        box-sizing: border-box;
    }

    /* ── Reset & background ── */
    .stApp { background-color: #F5F5F0 !important; }
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        overflow: hidden;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    section[data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"]  { display: none !important; }
    [data-testid="stDecoration"]      { display: none !important; }
    div[data-testid="stToolbar"]      { display: none !important; }

    /* ── SIDEBAR ── */
    .sidebar {
        position: fixed;
        left: 0; top: 0; bottom: 0;
        width: 64px;
        background: #1a1a1a;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.2rem 0;
        z-index: 999;
        gap: 0.2rem;
    }
    .sb-logo {
        width: 38px; height: 38px;
        background: #CC0000;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1rem; color: white;
        margin-bottom: 1.5rem;
    }
    .sb-icon {
        width: 40px; height: 40px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; cursor: pointer; color: #888;
        transition: all 0.15s;
    }
    .sb-icon:hover, .sb-icon.active { background: #2a2a2a; color: #fff; }
    .sb-divider { width: 30px; height: 1px; background: #2a2a2a; margin: 0.8rem 0; }

    /* ── MAIN WRAPPER ── */
    .dash-wrapper {
        margin-left: 64px;
        padding: 0;
        height: 100vh;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    /* ── TOP BAR ── */
    .topbar {
        background: #ffffff;
        padding: 0.75rem 1.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #EBEBEB;
        flex-shrink: 0;
    }
    .topbar-title { font-size: 1.15rem; font-weight: 700; color: #1a1a1a; margin: 0; }
    .topbar-sub   { font-size: 0.72rem; color: #999; margin: 0; }
    .topbar-right { display: flex; align-items: center; gap: 0.8rem; }
    .badge {
        background: #F5F5F0; border: 1px solid #EBEBEB;
        border-radius: 20px; padding: 0.3rem 0.9rem;
        font-size: 0.75rem; color: #666; font-weight: 500;
    }

    /* ── TAB ROW ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-bottom: 1px solid #EBEBEB;
        gap: 0; padding: 0 1.8rem;
        box-shadow: none;
        flex-shrink: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #aaa !important;
        font-weight: 600; font-size: 0.78rem; letter-spacing: 0.3px;
        border: none; border-bottom: 2px solid transparent;
        padding: 0.65rem 1.1rem; border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #CC0000 !important;
        border-bottom: 2px solid #CC0000 !important;
        background: transparent !important;
    }

    /* ── CONTENT AREA ── */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0 !important;
        overflow: hidden;
    }

    /* ── KPI ROW ── */
    .kpi-row {
        display: flex; gap: 1px;
        background: #EBEBEB;
        border-bottom: 1px solid #EBEBEB;
        flex-shrink: 0;
    }
    .kpi-box {
        flex: 1; background: #ffffff;
        padding: 0.9rem 1.8rem;
    }
    .kpi-label {
        font-size: 0.65rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.9px;
        color: #999; margin: 0 0 0.2rem 0;
    }
    .kpi-value { font-size: 1.55rem; font-weight: 700; color: #CC0000; margin: 0; line-height: 1.2; }
    .kpi-sub   { font-size: 0.68rem; color: #aaa; margin: 0.15rem 0 0 0; }

    /* ── CONTROL ROW ── */
    .ctrl-row {
        background: #ffffff;
        padding: 0.5rem 1.8rem;
        border-bottom: 1px solid #EBEBEB;
        display: flex; align-items: center; gap: 2rem;
        flex-shrink: 0;
    }
    .ctrl-info-label {
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.8px; color: #999; margin: 0;
    }
    .ctrl-info-val {
        font-size: 0.82rem; font-weight: 600; color: #1a1a1a; margin: 0.1rem 0 0 0;
    }
    .ctrl-info-mape { font-size: 0.78rem; font-weight: 700; color: #CC0000; margin: 0; }

    /* ── Selectbox tweak ── */
    div[data-testid="stSelectbox"] label {
        font-size: 0.65rem !important; font-weight: 700 !important;
        text-transform: uppercase !important; letter-spacing: 0.9px !important;
        color: #999 !important; margin-bottom: 0.15rem !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 8px !important;
        border-color: #EBEBEB !important;
        background: #FAFAFA !important;
        font-size: 0.82rem !important;
        min-height: 36px !important;
    }

    /* ── CHART AREA ── */
    .charts-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1px;
        background: #EBEBEB;
        flex: 1;
        overflow: hidden;
    }
    .chart-panel {
        background: #ffffff;
        padding: 1rem 1.2rem 0.5rem 1.2rem;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    .chart-panel.wide {
        grid-column: span 2;
    }
    .panel-label {
        font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.9px; color: #CC0000; margin: 0 0 0.1rem 0;
    }
    .panel-title {
        font-size: 0.88rem; font-weight: 700; color: #1a1a1a; margin: 0 0 0.3rem 0;
    }

    /* ── XAI feature bars ── */
    .feat-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.55rem; }
    .feat-dir {
        width: 22px; height: 22px; border-radius: 6px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
    }
    .feat-name  { font-size: 0.75rem; font-weight: 600; color: #1a1a1a; }
    .feat-desc  { font-size: 0.65rem; color: #999; }
    .feat-val   { font-size: 0.65rem; color: #bbb; }
    .feat-bar-wrap { flex: 1; height: 5px; background: #F0F0EC; border-radius: 3px; }
    .feat-bar-fill { height: 5px; border-radius: 3px; }
    .feat-impact { font-size: 0.68rem; font-weight: 600; width: 44px; text-align: right; flex-shrink: 0; }

    /* ── District mini cards ── */
    .dist-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.6rem;
        padding: 0.8rem 1.2rem;
    }
    .dist-card {
        background: #FAFAFA;
        border-radius: 10px;
        padding: 0.7rem 0.8rem;
        border-top: 3px solid;
    }
    .dist-card-name { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #999; margin: 0; }
    .dist-card-val  { font-size: 1.1rem; font-weight: 700; margin: 0.15rem 0 0 0; }
    .dist-card-sub  { font-size: 0.62rem; color: #aaa; margin: 0.05rem 0 0 0; }
    .dist-card-mape { font-size: 0.65rem; font-weight: 600; color: #999; margin: 0.2rem 0 0 0; }

    /* ── Note box ── */
    .note-box {
        background: #FFF8F8; border-left: 3px solid #CC0000;
        padding: 0.6rem 0.8rem; border-radius: 0 6px 6px 0;
        font-size: 0.72rem; color: #666; margin-top: 0.5rem;
    }

    /* ── Table tweak ── */
    [data-testid="stDataFrame"] { border-radius: 8px; }
    [data-testid="stDataFrame"] table { font-size: 0.78rem !important; }

    /* ── Metric override ── */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #CC0000 !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #999 !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.8px; }

    /* force iframe height to fill window */
    .element-container iframe { height: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature dictionary ─────────────────────────────────────────────────────────
FEAT_LABEL = {
    'lag_1':       'lag_1 — Jumlah servis 1 minggu lalu',
    'lag_2':       'lag_2 — Jumlah servis 2 minggu lalu',
    'lag_4':       'lag_4 — Jumlah servis 4 minggu lalu (±1 bln)',
    'lag_8':       'lag_8 — Jumlah servis 8 minggu lalu (±2 bln)',
    'lag_13':      'lag_13 — Jumlah servis 13 minggu lalu (±3 bln)',
    'roll4_mean':  'roll4_mean — Rata-rata servis 4 minggu terakhir',
    'roll8_mean':  'roll8_mean — Rata-rata servis 8 minggu terakhir',
    'roll13_mean': 'roll13_mean — Rata-rata servis 13 minggu terakhir',
    'roll4_std':   'roll4_std — Volatilitas servis 4 minggu terakhir',
    'roll8_std':   'roll8_std — Volatilitas servis 8 minggu terakhir',
    'month':       'month — Bulan dalam tahun (1–12)',
    'quarter':     'quarter — Kuartal dalam tahun (1–4)',
    'year':        'year — Tahun',
    'month_sin':   'month_sin — Pola musiman bulan (sinus)',
    'month_cos':   'month_cos — Pola musiman bulan (kosinus)',
    'is_lebaran':  'is_lebaran — Indikator minggu Lebaran',
    'is_holiday':  'is_holiday — Indikator minggu libur nasional',
}

def feat_label(name):
    return FEAT_LABEL.get(name, name)

def feat_desc(name):
    label = FEAT_LABEL.get(name, name)
    return label.split(' — ', 1)[1] if ' — ' in label else label

FEAT_TABLE = pd.DataFrame([
    {'Nama Fitur': k, 'Penjelasan': feat_desc(k)}
    for k in FEAT_LABEL
])

def dist_label(raw):
    return raw.replace("KECAMATAN_", "District ")

# ── Load assets ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = "streamlit_assets"
    with open(f"{base}/lgb_models.pkl",       "rb") as f: lgb_models       = pickle.load(f)
    with open(f"{base}/forecast_results.pkl",  "rb") as f: forecast_results  = pickle.load(f)
    with open(f"{base}/shap_results.pkl",      "rb") as f: shap_results     = pickle.load(f)
    with open(f"{base}/config.pkl",            "rb") as f: config           = pickle.load(f)
    dfs_smooth    = pd.read_csv(f"{base}/dfs_smooth.csv")
    dfs_smooth['Tanggal Servis'] = pd.to_datetime(dfs_smooth['Tanggal Servis'])
    df_comparison = pd.read_csv(f"{base}/model_comparison.csv")
    df_validation = pd.read_csv(f"{base}/validation_metrics.csv")
    for kec in forecast_results:
        forecast_results[kec]['ds'] = pd.to_datetime(forecast_results[kec]['ds'])
    return lgb_models, forecast_results, shap_results, config, dfs_smooth, df_comparison, df_validation

lgb_models, forecast_results, shap_results, config, dfs_smooth, df_comparison, df_validation = load_assets()

TOP5          = config['TOP5']
LEBARAN       = config['LEBARAN']
KEC_COLORS    = config['KEC_COLORS']
FEAT_COLS     = config.get('FEAT_COLS', [
    'lag_1','lag_2','lag_4','lag_8','lag_13',
    'roll4_mean','roll8_mean','roll13_mean','roll4_std','roll8_std',
    'month','quarter','year','month_sin','month_cos','is_lebaran','is_holiday'
])
KEC_COLOR_MAP = dict(zip(TOP5, KEC_COLORS))
LEBARAN_STR   = {yr: v[0] for yr, v in LEBARAN.items()}

LEGEND = dict(
    orientation='h', yanchor='top', y=-0.22, xanchor='center', x=0.5,
    bgcolor='rgba(255,255,255,0.95)', bordercolor='#EBEBEB', borderwidth=1,
    font=dict(size=9, color='#1a1a1a', family='Inter'),
)

def base_layout(**kwargs):
    d = dict(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        font=dict(color='#1a1a1a', family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='#F5F5F0', linecolor='#EBEBEB',
                   tickfont=dict(color='#aaa', family='Inter', size=10)),
        yaxis=dict(showgrid=True, gridcolor='#F5F5F0', linecolor='#EBEBEB',
                   tickfont=dict(color='#aaa', family='Inter', size=10)),
        margin=dict(l=10, r=10, t=10, b=80),
        legend=LEGEND,
    )
    d.update(kwargs)
    return d

def make_forecast_chart(kec, sel_idx=None):
    fc    = forecast_results[kec]
    hist  = dfs_smooth[dfs_smooth['Kecamatan Bengkel'] == kec].tail(26)
    color = KEC_COLOR_MAP[kec]
    fc_ds = fc['ds'].astype(str)
    fig   = go.Figure()
    fig.add_trace(go.Bar(
        x=hist['Tanggal Servis'].astype(str), y=hist['Jumlah Servis'],
        name='Historis 2025', marker_color='#dbeafe', marker_line_width=0, opacity=0.9))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_ds, fc_ds[::-1]]),
        y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
        fill='toself', fillcolor='rgba(204,0,0,0.07)',
        line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
    fig.add_trace(go.Scatter(x=fc_ds, y=fc['pessimistic'], mode='lines',
        name='Pesimistis -16%', line=dict(color='#ccc', dash='dash', width=1.2)))
    fig.add_trace(go.Scatter(x=fc_ds, y=fc['optimistic'], mode='lines',
        name='Optimistis +18%', line=dict(color='#16a34a', dash='dash', width=1.2)))
    fig.add_trace(go.Scatter(x=fc_ds, y=fc['forecast'], mode='lines',
        name='Prakiraan (LightGBM)', line=dict(color=color, width=2.2)))
    lb26 = LEBARAN_STR.get(2026)
    if lb26:
        fig.add_shape(type='line', x0=lb26, x1=lb26, y0=0, y1=1, yref='paper',
                      line=dict(color='#16a34a', width=1.2, dash='dash'))
        fig.add_annotation(x=lb26, y=1.04, yref='paper', text='Lebaran',
                           showarrow=False, font=dict(size=8, color='#16a34a', family='Inter'),
                           xanchor='center')
    if sel_idx is not None:
        fig.add_shape(type='line',
                      x0=str(fc['ds'].iloc[sel_idx]), x1=str(fc['ds'].iloc[sel_idx]),
                      y0=0, y1=1, yref='paper',
                      line=dict(color='#CC0000', width=1.2, dash='dot'))
    fig.update_layout(**base_layout(
        height=260,
        xaxis=dict(tickformat='%b %Y', showgrid=True, gridcolor='#F5F5F0',
                   linecolor='#EBEBEB', tickfont=dict(color='#aaa', family='Inter', size=9)),
        yaxis=dict(title='Servis/Minggu', showgrid=True, gridcolor='#F5F5F0',
                   linecolor='#EBEBEB', tickfont=dict(color='#aaa', family='Inter', size=9),
                   title_font=dict(size=9, color='#bbb')),
    ))
    return fig

def make_shap_bar(kec):
    sr        = shap_results[kec]
    mean_shap = np.abs(sr['shap_values']).mean(axis=0)
    labels    = [feat_label(f) for f in sr['feat_cols']]
    df        = pd.DataFrame({'Fitur': labels, 'Kepentingan': mean_shap}).sort_values('Kepentingan')
    fig = go.Figure(go.Bar(
        x=df['Kepentingan'], y=df['Fitur'], orientation='h',
        marker_color='#CC0000', marker_line_width=0, opacity=0.8))
    fig.update_layout(**base_layout(
        height=290, margin=dict(l=10, r=10, t=5, b=30),
        xaxis=dict(title='Rata-rata |SHAP|', showgrid=True, gridcolor='#F5F5F0',
                   linecolor='#EBEBEB', tickfont=dict(color='#aaa', family='Inter', size=8),
                   title_font=dict(size=9, color='#bbb')),
        yaxis=dict(showgrid=False, tickfont=dict(color='#555', family='Inter', size=8)),
        legend=dict(orientation='h', y=-0.05),
    ))
    return fig

def make_comparison_chart(df_comparison):
    model_colors = {'XGBoost': '#E6A817', 'LightGBM': '#CC0000', 'Stacking': '#639922'}
    fig = go.Figure()
    for model_name in df_comparison['Model'].unique():
        sub = df_comparison[df_comparison['Model'] == model_name]
        fig.add_trace(go.Bar(
            name=model_name,
            x=[dist_label(k) for k in sub['Kecamatan']],
            y=sub['MAPE (%)'],
            marker_color=model_colors.get(model_name, '#999'),
            marker_line_width=0))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                  line=dict(color='#CC0000', width=1, dash='dash'))
    fig.update_layout(**base_layout(
        barmode='group', height=220,
        margin=dict(l=10, r=10, t=10, b=60),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F5F5F0',
                   linecolor='#EBEBEB', tickfont=dict(color='#aaa', family='Inter', size=9),
                   title_font=dict(size=9, color='#bbb')),
        xaxis=dict(showgrid=False, tickfont=dict(color='#555', family='Inter', size=10)),
    ))
    return fig

def make_val_chart(df_validation):
    fig = go.Figure(go.Bar(
        x=[dist_label(k) for k in df_validation['Kecamatan']],
        y=df_validation['MAPE (%)'],
        marker_color=[KEC_COLOR_MAP[k] for k in df_validation['Kecamatan']],
        marker_line_width=0,
        text=df_validation['MAPE (%)'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        textfont=dict(color='#555', family='Inter', size=10),
    ))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                  line=dict(color='#CC0000', width=1, dash='dash'))
    fig.update_layout(**base_layout(
        height=200, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=40),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F5F5F0',
                   linecolor='#EBEBEB', tickfont=dict(color='#aaa', family='Inter', size=9),
                   title_font=dict(size=9, color='#bbb')),
        xaxis=dict(showgrid=False, tickfont=dict(color='#555', family='Inter', size=10)),
    ))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# RENDER SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sidebar">
  <div class="sb-logo">A</div>
  <div class="sb-icon active">📊</div>
  <div class="sb-icon">🧠</div>
  <div class="sb-icon">📋</div>
  <div class="sb-icon">✅</div>
  <div class="sb-divider"></div>
  <div class="sb-icon">⚙️</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-left:64px">
<div class="topbar">
  <div>
    <p class="topbar-title">AHASS Prakiraan Permintaan</p>
    <p class="topbar-sub">Prakiraan Permintaan Servis Mingguan · Top 5 District Jakarta · LightGBM + SHAP XAI</p>
  </div>
  <div class="topbar-right">
    <span class="badge">📅 Prakiraan 2026–2027</span>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONTROLS (inline, no card border)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-left:64px; background:#fff; border-bottom:1px solid #EBEBEB; padding: 0.4rem 1.8rem;">', unsafe_allow_html=True)
cc1, cc2, cc3 = st.columns([2, 4, 2])
with cc1:
    selected_kec = st.selectbox("District", options=TOP5, format_func=dist_label, index=0, key="sel_kec")
with cc2:
    fc_data = forecast_results[selected_kec]
    week_options = [f"Minggu {i+1} — {d.strftime('%d %b %Y')}" for i, d in enumerate(fc_data['ds'])]
    selected_week_label = st.selectbox("Minggu Prakiraan", options=week_options, index=0, key="sel_week")
    selected_week_idx   = week_options.index(selected_week_label)
with cc3:
    st.markdown(
        f"<p class='ctrl-info-label'>Model Terbaik</p>"
        f"<p class='ctrl-info-val'>LightGBM &nbsp;·&nbsp; Pelatihan 2022–2024 · Uji 2025</p>"
        f"<p class='ctrl-info-mape'>MAPE Uji {dist_label(selected_kec)}: {lgb_models[selected_kec]['mape']*100:.1f}%</p>",
        unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-left:64px">', unsafe_allow_html=True)

fc_row = fc_data.iloc[selected_week_idx]

tab1, tab2, tab3, tab4 = st.tabs(["  PRAKIRAAN  ", "  PENJELASAN XAI  ", "  PERBANDINGAN MODEL  ", "  VALIDASI 2026  "])

# ═══ TAB 1 — PRAKIRAAN ═══════════════════════════════════════════════════════
with tab1:
    # KPI strip
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-box">
        <p class="kpi-label">District</p>
        <p class="kpi-value" style="font-size:1.2rem;color:#1a1a1a">{dist_label(selected_kec)}</p>
      </div>
      <div class="kpi-box">
        <p class="kpi-label">Prakiraan Dasar</p>
        <p class="kpi-value">{fc_row['forecast']:,.0f}</p>
        <p class="kpi-sub">servis / minggu</p>
      </div>
      <div class="kpi-box">
        <p class="kpi-label">Optimistis +18%</p>
        <p class="kpi-value" style="color:#16a34a">{fc_row['optimistic']:,.0f}</p>
      </div>
      <div class="kpi-box">
        <p class="kpi-label">Pesimistis -16%</p>
        <p class="kpi-value" style="color:#9ca3af">{fc_row['pessimistic']:,.0f}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Two-panel grid: chart left (wide), district cards right
    col_chart, col_cards = st.columns([2, 1])

    with col_chart:
        st.markdown('<p class="panel-label" style="padding:0.8rem 0 0 0">Prakiraan Permintaan</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-title" style="padding:0 0 0.3rem 0">Grafik Prakiraan 52 Minggu</p>', unsafe_allow_html=True)
        st.plotly_chart(make_forecast_chart(selected_kec, selected_week_idx),
                        use_container_width=True, config={'displayModeBar': False})

    with col_cards:
        st.markdown('<p class="panel-label" style="padding:0.8rem 0 0.4rem 0">Semua District</p>', unsafe_allow_html=True)
        for kec, color in zip(TOP5, KEC_COLORS):
            fc_kec = forecast_results[kec]
            ann = fc_kec['forecast'].mean()
            tot = fc_kec['forecast'].sum()
            mape = lgb_models[kec]['mape'] * 100
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.7rem;padding:0.45rem 0;
                        border-bottom:1px solid #F5F5F0;">
              <div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0"></div>
              <div style="flex:1">
                <span style="font-size:0.75rem;font-weight:700;color:#1a1a1a">{dist_label(kec)}</span>
                <span style="font-size:0.68rem;color:#aaa;margin-left:0.4rem">MAPE {mape:.1f}%</span>
              </div>
              <div style="text-align:right">
                <p style="font-size:0.82rem;font-weight:700;color:{color};margin:0">{ann:,.0f}</p>
                <p style="font-size:0.62rem;color:#bbb;margin:0">∑ {tot:,.0f}/thn</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ═══ TAB 2 — XAI ═════════════════════════════════════════════════════════════
with tab2:
    fc_row_xai = forecast_results[selected_kec].iloc[selected_week_idx]
    sr      = shap_results[selected_kec]
    idx     = min(selected_week_idx, len(sr['shap_values']) - 1)
    sv      = sr['shap_values'][idx]
    xv      = sr['X_test'][idx]
    shap_df = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
    shap_df['abs_shap'] = shap_df['shap'].abs()
    shap_df_top = shap_df.sort_values('abs_shap', ascending=False).head(5)
    max_abs = shap_df_top['abs_shap'].max()

    col_a, col_b, col_c = st.columns([1, 1.4, 1.6])

    with col_a:
        st.markdown(f"""
        <p class="panel-label" style="padding:0.8rem 0 0.1rem 0">Explainable AI</p>
        <p class="panel-title">Mengapa Prakiraan Ini?</p>
        <p style="font-size:0.7rem;color:#aaa;margin:0 0 0.6rem 0">
          {fc_row_xai['ds'].strftime('%d %b %Y')} · {dist_label(selected_kec)}<br>
          Prakiraan: <strong style="color:#CC0000">{fc_row_xai['forecast']:,.0f}</strong> servis/minggu
        </p>
        <p style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;
                  color:#aaa;margin:0 0 0.5rem 0">5 Fitur Paling Berpengaruh</p>
        """, unsafe_allow_html=True)

        for _, row in shap_df_top.iterrows():
            is_pos  = row['shap'] > 0
            clr     = "#CC0000" if is_pos else "#2563eb"
            bg      = "#FFF0F0" if is_pos else "#EFF6FF"
            arrow   = "▲" if is_pos else "▼"
            bar_pct = int((row['abs_shap'] / max_abs) * 100)
            st.markdown(f"""
            <div class="feat-row">
              <div class="feat-dir" style="background:{bg};color:{clr}">{arrow}</div>
              <div style="flex:1;min-width:0">
                <div style="display:flex;align-items:baseline;gap:0.3rem">
                  <span class="feat-name">{row['feature']}</span>
                  <span class="feat-val">= {row['value']:.1f}</span>
                </div>
                <div class="feat-desc">{feat_desc(row['feature'])}</div>
                <div class="feat-bar-wrap" style="margin-top:0.25rem">
                  <div class="feat-bar-fill" style="width:{bar_pct}%;background:{clr};opacity:0.7"></div>
                </div>
              </div>
              <span class="feat-impact" style="color:{clr}">{row['shap']:+.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""<div class="note-box">
          ▲ Merah = menaikkan prakiraan &nbsp;·&nbsp; ▼ Biru = menurunkan prakiraan
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<p class="panel-label" style="padding:0.8rem 0 0.1rem 0">Kepentingan Global</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="panel-title">{dist_label(selected_kec)}</p>', unsafe_allow_html=True)
        st.plotly_chart(make_shap_bar(selected_kec), use_container_width=True,
                        config={'displayModeBar': False})

    with col_c:
        st.markdown('<p class="panel-label" style="padding:0.8rem 0 0.1rem 0">Semua District</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">Kepentingan SHAP Perbandingan</p>', unsafe_allow_html=True)
        fig_all = go.Figure()
        for kec, color in zip(TOP5, KEC_COLORS):
            sr2       = shap_results[kec]
            mean_shap = np.abs(sr2['shap_values']).mean(axis=0)
            labels    = [feat_label(f) for f in sr2['feat_cols']]
            fig_all.add_trace(go.Bar(name=dist_label(kec), x=labels, y=mean_shap,
                                     marker_color=color, marker_line_width=0))
        fig_all.update_layout(**base_layout(
            barmode='group', height=290,
            margin=dict(l=10, r=10, t=5, b=120),
            xaxis=dict(tickangle=-35, showgrid=False, tickfont=dict(color='#888', family='Inter', size=8)),
            yaxis=dict(title='|SHAP|', showgrid=True, gridcolor='#F5F5F0',
                       tickfont=dict(color='#aaa', family='Inter', size=8),
                       title_font=dict(size=9, color='#bbb')),
        ))
        st.plotly_chart(fig_all, use_container_width=True, config={'displayModeBar': False})

        st.markdown('<p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#aaa;margin:0.4rem 0 0.3rem 0">Kamus Fitur</p>', unsafe_allow_html=True)
        st.dataframe(FEAT_TABLE, use_container_width=True, hide_index=True, height=160)

# ═══ TAB 3 — PERBANDINGAN MODEL ══════════════════════════════════════════════
with tab3:
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<p class="panel-label" style="padding:0.8rem 0 0.1rem 0">Evaluasi Model</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">MAPE per Model dan District</p>', unsafe_allow_html=True)
        st.plotly_chart(make_comparison_chart(df_comparison), use_container_width=True,
                        config={'displayModeBar': False})

        st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#999;margin:0.3rem 0 0.3rem 0">Tabel Hasil Lengkap</p>', unsafe_allow_html=True)
        df_comp_disp = df_comparison.copy()
        df_comp_disp.insert(0, 'District', df_comp_disp['Kecamatan'].apply(dist_label))
        df_comp_disp = df_comp_disp[['Model','District','MAPE (%)','RMSE','MAE','R2']].sort_values(['District','MAPE (%)'])
        st.dataframe(df_comp_disp, use_container_width=True, height=200, hide_index=True)

    with col_right:
        st.markdown('<p class="panel-label" style="padding:0.8rem 0 0.1rem 0">Rata-rata Kinerja</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">Semua District</p>', unsafe_allow_html=True)

        avg_rows = []
        for model_name in df_comparison['Model'].unique():
            sub = df_comparison[df_comparison['Model'] == model_name]
            avg_rows.append({
                'Model': model_name,
                'Rata-rata MAPE (%)': round(sub['MAPE (%)'].mean(), 1),
                'Rata-rata RMSE':     int(sub['RMSE'].mean()),
                'Rata-rata MAE':      int(sub['MAE'].mean()),
                'Rata-rata R²':       round(sub['R2'].mean(), 3),
            })
        df_avg = pd.DataFrame(avg_rows).sort_values('Rata-rata MAPE (%)')
        st.dataframe(df_avg.style.highlight_min(subset=['Rata-rata MAPE (%)'], color='#fff0f0'),
                     use_container_width=True, height=180, hide_index=True)

        st.markdown('<p class="panel-label" style="padding:0.6rem 0 0.1rem 0">Interpretasi</p>', unsafe_allow_html=True)
        st.success("**Performa Terbaik:** LightGBM secara konsisten lebih unggul dibanding XGBoost dan Stacking di mayoritas district.")
        st.markdown('<p style="font-size:0.72rem;color:#999;margin:0.4rem 0 0 0">Garis merah putus-putus = target MAPE 15%</p>', unsafe_allow_html=True)

# ═══ TAB 4 — VALIDASI 2026 ════════════════════════════════════════════════════
with tab4:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<p class="panel-label" style="padding:0.8rem 0 0.1rem 0">Evaluasi Luar Sampel</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">Validasi 2026 (Jan–Mar)</p>', unsafe_allow_html=True)
        st.caption("Validasi vs data aktual 2026. Minggu hilang (9–23 Feb) diisi mean. Lebaran dilaporkan terpisah.")
        st.plotly_chart(make_val_chart(df_validation), use_container_width=True,
                        config={'displayModeBar': False})

        df_val_disp = df_validation.copy()
        df_val_disp.insert(0, 'District', df_val_disp['Kecamatan'].apply(dist_label))
        df_val_disp = df_val_disp[['District','Model','MAPE (%)','RMSE','MAE','Bias','Weeks']].rename(columns={'Weeks':'Minggu'})
        st.dataframe(df_val_disp.style.highlight_min(subset=['MAPE (%)'], color='#fff0f0'),
                     use_container_width=True, height=200, hide_index=True)

    with col_r:
        st.markdown('<p class="panel-label" style="padding:0.8rem 0 0.1rem 0">Interpretasi Hasil</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">Temuan Utama</p>', unsafe_allow_html=True)
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        items = [
            ("✅", "#16a34a", "#F0FDF4", "Performa Baik",
             "District D dan E mencapai MAPE di bawah 12% selama periode permintaan normal."),
            ("⚠️", "#d97706", "#FFFBEB", "Galat Lebih Tinggi",
             "District A dan C menunjukkan galat lebih tinggi akibat volatilitas permintaan yang melebihi pola historis."),
            ("🔴", "#CC0000", "#FFF5F5", "Efek Lebaran",
             "MAPE 29–75% selama minggu Lebaran. Penurunan permintaan hari raya berada di luar rentang pelatihan model."),
        ]
        for icon, clr, bg, title, desc in items:
            st.markdown(f"""
            <div style="background:{bg};border-left:3px solid {clr};border-radius:0 10px 10px 0;
                        padding:0.8rem 1rem;margin-bottom:0.7rem">
              <p style="font-size:0.82rem;font-weight:700;color:{clr};margin:0">{icon} {title}</p>
              <p style="font-size:0.75rem;color:#555;margin:0.2rem 0 0 0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
