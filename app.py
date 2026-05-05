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

html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ── */
.stApp { background-color: #EAECF0 !important; }
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }
[data-testid="stDecoration"]      { display: none !important; }
[data-testid="stStatusWidget"]    { display: none !important; }
.stDeployButton                   { display: none !important; }

/* ── SIDEBAR (fixed left) ── */
.sidebar {
    position: fixed; left: 0; top: 0; bottom: 0; width: 64px;
    background: #2D2D2D; display: flex; flex-direction: column;
    align-items: center; padding: 18px 0; z-index: 9999; gap: 4px;
}
.sb-logo {
    width: 40px; height: 40px; background: #CC0000; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1rem; color: #fff; margin-bottom: 18px;
}
.sb-icon {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; color: #777; cursor: pointer; transition: 0.15s;
}
.sb-icon:hover { background: #3a3a3a; color: #fff; }
.sb-icon.on    { background: #CC0000; color: #fff; }
.sb-sep { width: 28px; height: 1px; background: #3a3a3a; margin: 8px 0; }
.sb-label {
    font-size: 0.48rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px; color: #555; margin: 6px 0 2px 0;
}
.sb-bottom { margin-top: auto; display: flex; flex-direction: column; align-items: center; gap: 4px; }

/* ── MAIN wrapper ── */
.main-wrap {
    margin-left: 64px;
    padding: 22px 24px 24px 24px;
    min-height: 100vh;
}

/* ── Page header ── */
.page-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 18px;
}
.page-title { font-size: 1.65rem; font-weight: 700; color: #1a1a1a; margin: 0; line-height: 1.2; }
.page-sub   { font-size: 0.75rem; color: #888; margin: 3px 0 0 0; }
.page-right { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
.date-badge {
    display: flex; align-items: center; gap: 6px;
    background: #fff; border: 1px solid #E0E2E7; border-radius: 8px;
    padding: 6px 14px; font-size: 0.74rem; color: #555; font-weight: 500;
}
.filter-btn {
    padding: 6px 16px; border-radius: 8px; border: 1.5px solid #E0E2E7;
    background: #fff; font-size: 0.74rem; font-weight: 600; color: #555; cursor: pointer;
}
.filter-btn.active { background: #CC0000; color: #fff; border-color: #CC0000; }

/* ── KPI CARDS ── */
.kpi-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin-bottom: 16px;
}
.kpi-card {
    background: #fff; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
    display: flex; flex-direction: column; gap: 4px;
}
.kpi-label {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #9CA3AF; margin: 0;
}
.kpi-delta { font-size: 0.72rem; font-weight: 600; color: #16a34a; }
.kpi-delta.down { color: #CC0000; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #1a1a1a; margin: 0; line-height: 1.1; }
.kpi-sub   { font-size: 0.65rem; color: #9CA3AF; margin: 0; }

/* ── CHART CARDS ── */
.chart-card {
    background: #fff; border-radius: 14px; padding: 16px 18px 10px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
}
.chart-label {
    font-size: 0.7rem; color: #9CA3AF; font-weight: 500; margin: 0 0 1px 0;
}
.chart-title {
    font-size: 0.88rem; font-weight: 700; color: #1a1a1a; margin: 0 0 10px 0;
}

/* ── Controls card ── */
.ctrl-card {
    background: #fff; border-radius: 14px; padding: 12px 18px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
    border-left: 4px solid #CC0000;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff; border-radius: 14px 14px 0 0;
    border-bottom: 1px solid #F0F1F3; padding: 0 6px; gap: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #9CA3AF !important;
    font-weight: 600; font-size: 0.78rem; letter-spacing: 0.2px;
    border: none; border-bottom: 2px solid transparent;
    padding: 12px 16px; border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #CC0000 !important; border-bottom: 2px solid #CC0000 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── Tab content wrapper ── */
.tab-content {
    background: #fff; border-radius: 0 0 14px 14px;
    padding: 18px 18px 20px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}

/* ── Inner grid ── */
.inner-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.inner-grid-2 { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; }

/* ── Metrics override ── */
[data-testid="metric-container"] {
    background: #fff; border-radius: 14px; padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important; font-weight: 700 !important; color: #CC0000 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.62rem !important; font-weight: 700 !important; color: #9CA3AF !important;
    text-transform: uppercase; letter-spacing: 1px;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] label {
    font-size: 0.62rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important; color: #9CA3AF !important;
}
div[data-testid="stSelectbox"] > div > div {
    border-radius: 9px !important; border-color: #E0E2E7 !important;
    background: #F9FAFB !important; font-size: 0.82rem !important; color: #1a1a1a !important;
}

/* ── District list ── */
.dist-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #F3F4F6;
}
.dist-dot  { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.dist-name { font-size: 0.78rem; font-weight: 600; color: #1a1a1a; }
.dist-mape { font-size: 0.64rem; color: #9CA3AF; margin-left: 4px; }
.dist-right { margin-left: auto; text-align: right; }
.dist-avg  { font-size: 0.84rem; font-weight: 700; margin: 0; }
.dist-sum  { font-size: 0.62rem; color: #9CA3AF; margin: 0; }

/* ── XAI feature row ── */
.feat-row  { display: flex; align-items: center; gap: 8px; margin-bottom: 9px; }
.feat-pill {
    width: 22px; height: 22px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 700;
}
.feat-body { flex: 1; min-width: 0; }
.feat-nm   { font-size: 0.75rem; font-weight: 700; color: #1a1a1a; }
.feat-vl   { font-size: 0.62rem; color: #9CA3AF; margin-left: 4px; }
.feat-desc { font-size: 0.63rem; color: #9CA3AF; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.feat-bar  { height: 4px; background: #F3F4F6; border-radius: 2px; margin-top: 4px; }
.feat-fill { height: 4px; border-radius: 2px; }
.feat-imp  { font-size: 0.7rem; font-weight: 600; width: 44px; text-align: right; flex-shrink: 0; }

/* ── Note box ── */
.note-box {
    background: #FFF5F5; border-left: 3px solid #CC0000;
    padding: 8px 12px; border-radius: 0 8px 8px 0;
    font-size: 0.7rem; color: #6B7280; margin-top: 10px;
}

/* ── Interp card ── */
.interp-card {
    border-left: 3px solid; border-radius: 0 10px 10px 0;
    padding: 10px 14px; margin-bottom: 10px;
}
.interp-ttl  { font-size: 0.8rem; font-weight: 700; margin: 0 0 3px 0; }
.interp-body { font-size: 0.72rem; color: #6B7280; margin: 0; }

/* ── Section label ── */
.sec-lbl {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #9CA3AF; margin: 0 0 2px 0;
}
.sec-ttl { font-size: 0.9rem; font-weight: 700; color: #1a1a1a; margin: 0 0 12px 0; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Column padding fix ── */
[data-testid="column"] { padding: 0 5px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }
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
def feat_label(n): return FEAT_LABEL.get(n, n)
def feat_desc(n):
    l = FEAT_LABEL.get(n, n)
    return l.split(' — ', 1)[1] if ' — ' in l else l

FEAT_TABLE = pd.DataFrame([{'Nama Fitur': k, 'Penjelasan': feat_desc(k)} for k in FEAT_LABEL])
def dist_label(r): return r.replace("KECAMATAN_", "District ")

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
KEC_COLOR_MAP = dict(zip(TOP5, KEC_COLORS))
LEBARAN_STR   = {yr: v[0] for yr, v in LEBARAN.items()}

LEGEND = dict(orientation='h', yanchor='top', y=-0.22, xanchor='center', x=0.5,
              bgcolor='rgba(255,255,255,0.97)', bordercolor='#E5E7EB', borderwidth=1,
              font=dict(size=9, color='#6B7280', family='Inter'))

def base_layout(**kw):
    d = dict(
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        font=dict(color='#1a1a1a', family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='#F9FAFB', linecolor='#E5E7EB',
                   tickfont=dict(color='#9CA3AF', family='Inter', size=10)),
        yaxis=dict(showgrid=True, gridcolor='#F9FAFB', linecolor='#E5E7EB',
                   tickfont=dict(color='#9CA3AF', family='Inter', size=10)),
        margin=dict(l=10, r=10, t=10, b=90),
        legend=LEGEND,
    )
    d.update(kw)
    return d

def chart_forecast(kec, sel_idx=None):
    fc   = forecast_results[kec]
    hist = dfs_smooth[dfs_smooth['Kecamatan Bengkel'] == kec].tail(26)
    clr  = KEC_COLOR_MAP[kec]
    fds  = fc['ds'].astype(str)
    fig  = go.Figure()
    fig.add_trace(go.Bar(x=hist['Tanggal Servis'].astype(str), y=hist['Jumlah Servis'],
                         name='Historis 2025', marker_color='#DBEAFE', marker_line_width=0, opacity=0.9))
    fig.add_trace(go.Scatter(x=pd.concat([fds, fds[::-1]]),
                             y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
                             fill='toself', fillcolor='rgba(204,0,0,0.07)',
                             line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
    fig.add_trace(go.Scatter(x=fds, y=fc['pessimistic'], mode='lines',
                             name='Pesimistis -16%', line=dict(color='#D1D5DB', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=fds, y=fc['optimistic'], mode='lines',
                             name='Optimistis +18%', line=dict(color='#16a34a', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=fds, y=fc['forecast'], mode='lines',
                             name='Prakiraan (LightGBM)', line=dict(color=clr, width=2.2)))
    lb = LEBARAN_STR.get(2026)
    if lb:
        fig.add_shape(type='line', x0=lb, x1=lb, y0=0, y1=1, yref='paper',
                      line=dict(color='#16a34a', width=1.2, dash='dash'))
        fig.add_annotation(x=lb, y=1.05, yref='paper', text='Lebaran', showarrow=False,
                           font=dict(size=8, color='#16a34a', family='Inter'), xanchor='center')
    if sel_idx is not None:
        x0 = str(fc['ds'].iloc[sel_idx])
        fig.add_shape(type='line', x0=x0, x1=x0, y0=0, y1=1, yref='paper',
                      line=dict(color='#CC0000', width=1.2, dash='dot'))
    fig.update_layout(**base_layout(height=340,
        xaxis=dict(tickformat='%b %Y', showgrid=True, gridcolor='#F9FAFB',
                   linecolor='#E5E7EB', tickfont=dict(color='#9CA3AF', family='Inter', size=10)),
        yaxis=dict(title='Servis / Minggu', showgrid=True, gridcolor='#F9FAFB',
                   linecolor='#E5E7EB', tickfont=dict(color='#9CA3AF', family='Inter', size=10),
                   title_font=dict(size=10, color='#9CA3AF'))))
    return fig

def chart_shap_bar(kec):
    sr  = shap_results[kec]
    ms  = np.abs(sr['shap_values']).mean(axis=0)
    lbs = [feat_label(f) for f in sr['feat_cols']]
    df  = pd.DataFrame({'F': lbs, 'V': ms}).sort_values('V')
    fig = go.Figure(go.Bar(x=df['V'], y=df['F'], orientation='h',
                           marker_color='#CC0000', marker_line_width=0, opacity=0.8))
    fig.update_layout(**base_layout(height=340, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title='Rata-rata |SHAP|', showgrid=True, gridcolor='#F9FAFB',
                   linecolor='#E5E7EB', tickfont=dict(color='#9CA3AF', family='Inter', size=9),
                   title_font=dict(size=10, color='#9CA3AF')),
        yaxis=dict(showgrid=False, tickfont=dict(color='#374151', family='Inter', size=8.5)),
        legend=dict(orientation='h', y=-0.05)))
    return fig

def chart_shap_all():
    fig = go.Figure()
    for kec, clr in zip(TOP5, KEC_COLORS):
        sr  = shap_results[kec]
        ms  = np.abs(sr['shap_values']).mean(axis=0)
        lbs = [feat_label(f) for f in sr['feat_cols']]
        fig.add_trace(go.Bar(name=dist_label(kec), x=lbs, y=ms,
                             marker_color=clr, marker_line_width=0))
    fig.update_layout(**base_layout(barmode='group', height=320,
        margin=dict(l=10, r=10, t=10, b=170),
        xaxis=dict(tickangle=-38, showgrid=False, tickfont=dict(color='#6B7280', family='Inter', size=8.5)),
        yaxis=dict(title='|SHAP|', showgrid=True, gridcolor='#F9FAFB',
                   tickfont=dict(color='#9CA3AF', family='Inter', size=9),
                   title_font=dict(size=10, color='#9CA3AF'))))
    return fig

def chart_comp():
    mc  = {'XGBoost': '#E6A817', 'LightGBM': '#CC0000', 'Stacking': '#639922'}
    fig = go.Figure()
    for mn in df_comparison['Model'].unique():
        sub = df_comparison[df_comparison['Model'] == mn]
        fig.add_trace(go.Bar(name=mn, x=[dist_label(k) for k in sub['Kecamatan']],
                             y=sub['MAPE (%)'], marker_color=mc.get(mn,'#999'), marker_line_width=0))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                  line=dict(color='#CC0000', width=1, dash='dash'))
    fig.add_annotation(x=0.98, y=15, xref='paper', text='Target 15%', showarrow=False,
                       font=dict(size=8.5, color='#CC0000', family='Inter'),
                       xanchor='right', yanchor='bottom')
    fig.update_layout(**base_layout(barmode='group', height=300,
        margin=dict(l=10, r=10, t=10, b=70),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F9FAFB',
                   tickfont=dict(color='#9CA3AF', family='Inter', size=10),
                   title_font=dict(size=10, color='#9CA3AF')),
        xaxis=dict(showgrid=False, tickfont=dict(color='#374151', family='Inter', size=11))))
    return fig

def chart_val():
    fig = go.Figure(go.Bar(
        x=[dist_label(k) for k in df_validation['Kecamatan']],
        y=df_validation['MAPE (%)'],
        marker_color=[KEC_COLOR_MAP[k] for k in df_validation['Kecamatan']],
        marker_line_width=0,
        text=df_validation['MAPE (%)'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside', textfont=dict(color='#374151', family='Inter', size=10)))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                  line=dict(color='#CC0000', width=1, dash='dash'))
    fig.update_layout(**base_layout(height=260, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=50),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F9FAFB',
                   tickfont=dict(color='#9CA3AF', family='Inter', size=10),
                   title_font=dict(size=10, color='#9CA3AF')),
        xaxis=dict(showgrid=False, tickfont=dict(color='#374151', family='Inter', size=11))))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sidebar">
  <div class="sb-logo">A</div>
  <div class="sb-icon on">📊</div>
  <div class="sb-icon">🗂️</div>
  <div class="sb-icon">🧠</div>
  <div class="sb-icon">✅</div>
  <div class="sb-sep"></div>
  <div class="sb-label">Export</div>
  <div class="sb-icon">📄</div>
  <div class="sb-icon">🖼️</div>
  <div class="sb-bottom">
    <div class="sb-sep"></div>
    <div class="sb-icon">ℹ️</div>
    <div class="sb-icon">⚙️</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

# ── Page header ──
st.markdown("""
<div class="page-header">
  <div>
    <p class="page-title">AHASS Prakiraan Permintaan</p>
    <p class="page-sub">Prakiraan Permintaan Servis Mingguan · Top 5 District Jakarta · LightGBM + SHAP XAI</p>
  </div>
  <div class="page-right">
    <div class="date-badge">📅 &nbsp; Prakiraan 2026 – 2027</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ──
st.markdown('<div class="ctrl-card">', unsafe_allow_html=True)
cc1, cc2, cc3 = st.columns([2, 4, 2])
with cc1:
    selected_kec = st.selectbox("District", options=TOP5, format_func=dist_label, index=0)
with cc2:
    fc_data = forecast_results[selected_kec]
    week_options = [f"Minggu {i+1} — {d.strftime('%d %b %Y')}" for i, d in enumerate(fc_data['ds'])]
    sel_week_lbl = st.selectbox("Minggu Prakiraan", options=week_options, index=0)
    sel_week_idx = week_options.index(sel_week_lbl)
with cc3:
    mape_pct = lgb_models[selected_kec]['mape'] * 100
    st.markdown(
        f"<p style='font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#9CA3AF;margin:0'>Model Terbaik</p>"
        f"<p style='font-size:0.82rem;font-weight:600;color:#1a1a1a;margin:2px 0'>LightGBM · 2022–2024 · Uji 2025</p>"
        f"<p style='font-size:0.76rem;font-weight:700;color:#CC0000;margin:0'>MAPE {dist_label(selected_kec)}: {mape_pct:.1f}%</p>",
        unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── KPI Cards ──
fc_row = fc_data.iloc[sel_week_idx]
avg_forecast = np.mean([forecast_results[k]['forecast'].mean() for k in TOP5])

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <p class="kpi-label">District</p>
    <p class="kpi-value" style="font-size:1.25rem;color:#1a1a1a">{dist_label(selected_kec)}</p>
    <p class="kpi-sub">periode aktif</p>
  </div>
  <div class="kpi-card">
    <p class="kpi-label">Prakiraan Dasar &nbsp;<span class="kpi-delta">▲ vs rata-rata</span></p>
    <p class="kpi-value" style="color:#CC0000">{fc_row['forecast']:,.0f}</p>
    <p class="kpi-sub">servis / minggu</p>
  </div>
  <div class="kpi-card">
    <p class="kpi-label">Optimistis +18% &nbsp;<span class="kpi-delta">▲</span></p>
    <p class="kpi-value" style="color:#16a34a">{fc_row['optimistic']:,.0f}</p>
    <p class="kpi-sub">skenario terbaik</p>
  </div>
  <div class="kpi-card">
    <p class="kpi-label">Pesimistis -16% &nbsp;<span class="kpi-delta down">▼</span></p>
    <p class="kpi-value" style="color:#6B7280">{fc_row['pessimistic']:,.0f}</p>
    <p class="kpi-sub">skenario terburuk</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["  PRAKIRAAN  ", "  PENJELASAN XAI  ", "  PERBANDINGAN MODEL  ", "  VALIDASI 2026  "])

# ═══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    col_chart, col_side = st.columns([3, 1])

    with col_chart:
        st.markdown('<p class="chart-label">Prakiraan Permintaan</p>', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Grafik Prakiraan 52 Minggu</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_forecast(selected_kec, sel_week_idx),
                        use_container_width=True, config={'displayModeBar': False})

    with col_side:
        st.markdown('<p class="chart-label">Semua District</p>', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Ringkasan</p>', unsafe_allow_html=True)
        for kec, clr in zip(TOP5, KEC_COLORS):
            fk   = forecast_results[kec]
            mape = lgb_models[kec]['mape'] * 100
            st.markdown(f"""
            <div class="dist-row">
              <div class="dist-dot" style="background:{clr}"></div>
              <div>
                <span class="dist-name">{dist_label(kec)}</span>
                <span class="dist-mape">MAPE {mape:.1f}%</span>
              </div>
              <div class="dist-right">
                <p class="dist-avg" style="color:{clr}">{fk['forecast'].mean():,.0f}</p>
                <p class="dist-sum">∑ {fk['forecast'].sum():,.0f}/thn</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    fc_xai  = forecast_results[selected_kec].iloc[sel_week_idx]
    sr      = shap_results[selected_kec]
    idx     = min(sel_week_idx, len(sr['shap_values']) - 1)
    sv, xv  = sr['shap_values'][idx], sr['X_test'][idx]
    shap_df = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
    shap_df['abs_shap'] = shap_df['shap'].abs()
    top5_sh = shap_df.sort_values('abs_shap', ascending=False).head(5)
    max_abs = top5_sh['abs_shap'].max()

    col_a, col_b, col_c = st.columns([1, 1.3, 1.8])

    with col_a:
        st.markdown(f"""
        <p class="chart-label">Explainable AI</p>
        <p class="chart-title">Mengapa Prakiraan Ini?</p>
        <p style="font-size:0.7rem;color:#9CA3AF;margin:-8px 0 12px 0">
          {fc_xai['ds'].strftime('%d %b %Y')} · {dist_label(selected_kec)} ·
          <strong style="color:#CC0000">{fc_xai['forecast']:,.0f}</strong> servis/minggu
        </p>
        <p style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#9CA3AF;margin:0 0 10px 0">5 Fitur Paling Berpengaruh</p>
        """, unsafe_allow_html=True)

        for _, row in top5_sh.iterrows():
            pos = row['shap'] > 0
            clr = "#CC0000" if pos else "#2563eb"
            bg  = "#FFF0F0" if pos else "#EFF6FF"
            ar  = "▲" if pos else "▼"
            bp  = int((row['abs_shap'] / max_abs) * 100)
            st.markdown(f"""
            <div class="feat-row">
              <div class="feat-pill" style="background:{bg};color:{clr}">{ar}</div>
              <div class="feat-body">
                <div style="display:flex;align-items:baseline;gap:4px">
                  <span class="feat-nm">{row['feature']}</span>
                  <span class="feat-vl">= {row['value']:.1f}</span>
                </div>
                <div class="feat-desc">{feat_desc(row['feature'])}</div>
                <div class="feat-bar">
                  <div class="feat-fill" style="width:{bp}%;background:{clr};opacity:0.65"></div>
                </div>
              </div>
              <span class="feat-imp" style="color:{clr}">{row['shap']:+.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""<div class="note-box">
          ▲ Merah = menaikkan prakiraan &nbsp;·&nbsp; ▼ Biru = menurunkan prakiraan
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<p class="chart-label">Kepentingan Global</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="chart-title">{dist_label(selected_kec)}</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_shap_bar(selected_kec), use_container_width=True,
                        config={'displayModeBar': False})

    with col_c:
        st.markdown('<p class="chart-label">Semua District</p>', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Perbandingan SHAP</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_shap_all(), use_container_width=True,
                        config={'displayModeBar': False})
        st.markdown('<p style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#9CA3AF;margin:8px 0 4px 0">Kamus Fitur</p>', unsafe_allow_html=True)
        st.dataframe(FEAT_TABLE, use_container_width=True, hide_index=True, height=140)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<p class="chart-label">Evaluasi Model</p>', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">MAPE per Model dan District — Periode Uji 2025</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_comp(), use_container_width=True, config={'displayModeBar': False})
        df_cd = df_comparison.copy()
        df_cd.insert(0, 'District', df_cd['Kecamatan'].apply(dist_label))
        df_cd = df_cd[['Model','District','MAPE (%)','RMSE','MAE','R2']].sort_values(['District','MAPE (%)'])
        st.markdown('<p style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#9CA3AF;margin:8px 0 4px 0">Tabel Hasil Lengkap</p>', unsafe_allow_html=True)
        st.dataframe(df_cd, use_container_width=True, height=200, hide_index=True)

    with col_r:
        st.markdown('<p class="chart-label">Rata-rata Kinerja</p>', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Semua District</p>', unsafe_allow_html=True)
        avg_rows = []
        for mn in df_comparison['Model'].unique():
            sub = df_comparison[df_comparison['Model'] == mn]
            avg_rows.append({'Model': mn, 'Avg MAPE (%)': round(sub['MAPE (%)'].mean(), 1),
                             'Avg RMSE': int(sub['RMSE'].mean()), 'Avg MAE': int(sub['MAE'].mean()),
                             'Avg R²': round(sub['R2'].mean(), 3)})
        df_avg = pd.DataFrame(avg_rows).sort_values('Avg MAPE (%)')
        st.dataframe(df_avg.style.highlight_min(subset=['Avg MAPE (%)'], color='#FFF0F0'),
                     use_container_width=True, height=160, hide_index=True)

        st.markdown('<p style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#9CA3AF;margin:14px 0 8px 0">Interpretasi</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="interp-card" style="background:#F0FDF4;border-color:#16a34a">
          <p class="interp-ttl" style="color:#16a34a">✅ LightGBM Unggul</p>
          <p class="interp-body">Secara konsisten lebih baik dibanding XGBoost dan Stacking di mayoritas district.</p>
        </div>
        <p style="font-size:0.65rem;color:#9CA3AF;margin:4px 0 0 0">Garis merah putus-putus = target MAPE 15%</p>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<p class="chart-label">Validasi Luar Sampel</p>', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Validasi 2026 — Januari s.d. Maret</p>', unsafe_allow_html=True)
        st.caption("Vs data aktual 2026. Minggu hilang (9–23 Feb) diisi mean. Lebaran dilaporkan terpisah.")
        st.plotly_chart(chart_val(), use_container_width=True, config={'displayModeBar': False})
        dv = df_validation.copy()
        dv.insert(0, 'District', dv['Kecamatan'].apply(dist_label))
        dv = dv[['District','Model','MAPE (%)','RMSE','MAE','Bias','Weeks']].rename(columns={'Weeks':'Minggu'})
        st.markdown('<p style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#9CA3AF;margin:8px 0 4px 0">Metrik Validasi</p>', unsafe_allow_html=True)
        st.dataframe(dv.style.highlight_min(subset=['MAPE (%)'], color='#FFF0F0'),
                     use_container_width=True, height=200, hide_index=True)

    with col_r:
        st.markdown('<p class="chart-label">Interpretasi Hasil</p>', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Temuan Utama</p>', unsafe_allow_html=True)
        for clr, bg, title, desc in [
            ("#16a34a","#F0FDF4","✅ Performa Baik",
             "District D dan E mencapai MAPE di bawah 12% selama periode permintaan normal."),
            ("#D97706","#FFFBEB","⚠️ Galat Lebih Tinggi",
             "District A dan C menunjukkan galat lebih tinggi akibat volatilitas permintaan yang melebihi pola historis."),
            ("#CC0000","#FFF5F5","🔴 Efek Lebaran",
             "MAPE 29–75% selama minggu Lebaran. Penurunan permintaan hari raya di luar rentang pelatihan."),
        ]:
            st.markdown(f"""
            <div class="interp-card" style="background:{bg};border-color:{clr}">
              <p class="interp-ttl" style="color:{clr}">{title}</p>
              <p class="interp-body">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-wrap
