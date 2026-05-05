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

/* ── Hard reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }

html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4, button {
    font-family: 'Inter', sans-serif !important;
}

/* ── Kill ALL streamlit chrome & spacing ── */
#MainMenu, footer, header { visibility: hidden !important; display: none !important; }
section[data-testid="stSidebar"]   { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stStatusWidget"]     { display: none !important; }
.stDeployButton                    { display: none !important; }
iframe[title="streamlit_app"]      { border: none !important; }

/* ── Remove ALL default padding/margin from app shell ── */
.stApp {
    background: #F2F2EF !important;
    margin: 0 !important; padding: 0 !important;
    overflow: hidden !important;
}
.main { padding: 0 !important; margin: 0 !important; }
.main > div { padding: 0 !important; }
.block-container {
    padding: 0 !important; margin: 0 !important;
    max-width: 100% !important; min-width: 100% !important;
    width: 100% !important;
}
/* Kill the extra top padding Streamlit injects */
.block-container > div:first-child { padding-top: 0 !important; margin-top: 0 !important; }
section.main > div { padding-top: 0 !important; }

/* ── LAYOUT SHELL ── */
.app-shell {
    display: flex;
    height: 100vh;
    width: 100%;
    overflow: hidden;
    background: #F2F2EF;
}

/* ── SIDEBAR ── */
.app-sidebar {
    width: 56px;
    background: #1C1C1C;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 16px 0;
    flex-shrink: 0;
    gap: 6px;
    height: 100vh;
}
.sb-logo {
    width: 36px; height: 36px;
    background: #CC0000; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; font-weight: 800; color: #fff;
    margin-bottom: 16px;
}
.sb-icon {
    width: 36px; height: 36px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; color: #666; cursor: pointer;
}
.sb-icon:hover { background: #2a2a2a; color: #fff; }
.sb-icon.active { background: #CC0000; color: #fff; }
.sb-spacer { flex: 1; }

/* ── MAIN CONTENT ── */
.app-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    background: #F2F2EF;
}

/* ── TOP BAR ── */
.topbar {
    background: #fff;
    padding: 10px 20px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #E8E8E5;
    flex-shrink: 0;
}
.topbar-left h1 {
    font-size: 1rem; font-weight: 700; color: #1a1a1a;
    margin: 0; line-height: 1.2;
}
.topbar-left p { font-size: 0.68rem; color: #aaa; margin: 1px 0 0 0; }
.topbar-badge {
    background: #F2F2EF; border: 1px solid #E8E8E5;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.7rem; color: #888; font-weight: 500;
}

/* ── CONTROL STRIP ── */
.ctrl-strip {
    background: #fff;
    border-bottom: 1px solid #E8E8E5;
    padding: 6px 20px;
    flex-shrink: 0;
    display: flex; align-items: center; gap: 16px;
}

/* ── Selectbox compact ── */
div[data-testid="stSelectbox"] {
    margin: 0 !important; padding: 0 !important;
}
div[data-testid="stSelectbox"] label {
    font-size: 0.6rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    color: #aaa !important; margin-bottom: 2px !important;
    line-height: 1 !important;
}
div[data-testid="stSelectbox"] > div > div {
    border-radius: 7px !important; border-color: #E8E8E5 !important;
    background: #FAFAF8 !important; font-size: 0.78rem !important;
    min-height: 30px !important; padding: 3px 8px !important;
    color: #1a1a1a !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff !important;
    border-bottom: 1px solid #E8E8E5 !important;
    padding: 0 20px !important; gap: 0 !important;
    flex-shrink: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #bbb !important;
    font-weight: 600 !important; font-size: 0.72rem !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    padding: 8px 14px !important; border-radius: 0 !important;
    letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    color: #CC0000 !important;
    border-bottom: 2px solid #CC0000 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 0 !important; overflow: hidden !important;
}
/* Hide the tab panel border */
[data-baseweb="tab-border"] { display: none !important; }

/* ── KPI STRIP ── */
.kpi-strip {
    display: flex; background: #fff;
    border-bottom: 1px solid #E8E8E5;
    flex-shrink: 0;
}
.kpi-cell {
    flex: 1; padding: 10px 20px;
    border-right: 1px solid #E8E8E5;
}
.kpi-cell:last-child { border-right: none; }
.kpi-lbl {
    font-size: 0.58rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #bbb; margin: 0 0 2px 0;
}
.kpi-val { font-size: 1.4rem; font-weight: 700; margin: 0; line-height: 1.1; }
.kpi-sub { font-size: 0.62rem; color: #bbb; margin: 2px 0 0 0; }

/* ── BODY GRID ── */
.body-grid {
    display: flex; flex: 1; overflow: hidden; gap: 1px;
    background: #E8E8E5;
}
.panel {
    background: #fff; overflow: hidden;
    display: flex; flex-direction: column;
    padding: 12px 16px 4px 16px;
}
.panel.grow { flex: 1; }
.panel-lbl {
    font-size: 0.58rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #CC0000; margin: 0 0 1px 0;
}
.panel-ttl {
    font-size: 0.82rem; font-weight: 700; color: #1a1a1a; margin: 0 0 6px 0;
}

/* ── District list rows ── */
.dist-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0; border-bottom: 1px solid #F2F2EF;
}
.dist-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dist-name { font-size: 0.75rem; font-weight: 600; color: #1a1a1a; }
.dist-mape { font-size: 0.65rem; color: #bbb; margin-left: 4px; }
.dist-nums { margin-left: auto; text-align: right; }
.dist-avg  { font-size: 0.82rem; font-weight: 700; }
.dist-sum  { font-size: 0.6rem; color: #bbb; }

/* ── XAI feature rows ── */
.feat-row { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
.feat-pill {
    width: 20px; height: 20px; border-radius: 5px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 700;
}
.feat-body { flex: 1; min-width: 0; }
.feat-top  { display: flex; align-items: baseline; gap: 4px; }
.feat-nm   { font-size: 0.72rem; font-weight: 700; color: #1a1a1a; }
.feat-vl   { font-size: 0.6rem; color: #bbb; }
.feat-desc-txt { font-size: 0.62rem; color: #999; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.feat-bar  { height: 4px; background: #F2F2EF; border-radius: 2px; margin-top: 3px; }
.feat-fill { height: 4px; border-radius: 2px; }
.feat-imp  { font-size: 0.68rem; font-weight: 600; width: 40px; text-align: right; flex-shrink: 0; }

/* ── Note ── */
.note {
    background: #FFF8F8; border-left: 3px solid #CC0000;
    padding: 6px 10px; border-radius: 0 6px 6px 0;
    font-size: 0.68rem; color: #777; margin-top: 6px;
}

/* ── Interp card ── */
.interp-card {
    border-left: 3px solid; border-radius: 0 8px 8px 0;
    padding: 8px 12px; margin-bottom: 8px;
}
.interp-title { font-size: 0.78rem; font-weight: 700; margin: 0 0 2px 0; }
.interp-body  { font-size: 0.7rem; color: #555; margin: 0; }

/* ── Streamlit column gap remove ── */
[data-testid="column"] { padding: 0 !important; }
div[data-testid="stHorizontalBlock"] { gap: 1px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

/* ── Remove plotly chart container padding ── */
[data-testid="stPlotlyChart"] { padding: 0 !important; margin: 0 !important; }
.js-plotly-plot { margin: 0 !important; }
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

LEGEND = dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5,
              bgcolor='rgba(255,255,255,0.95)', bordercolor='#E8E8E5', borderwidth=1,
              font=dict(size=8.5, color='#555', family='Inter'))

def base_layout(**kw):
    d = dict(
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        font=dict(color='#1a1a1a', family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='#F5F5F2', linecolor='#E8E8E5',
                   tickfont=dict(color='#bbb', family='Inter', size=9)),
        yaxis=dict(showgrid=True, gridcolor='#F5F5F2', linecolor='#E8E8E5',
                   tickfont=dict(color='#bbb', family='Inter', size=9)),
        margin=dict(l=8, r=8, t=6, b=72),
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
                         name='Historis 2025', marker_color='#dbeafe', marker_line_width=0, opacity=0.9))
    fig.add_trace(go.Scatter(x=pd.concat([fds, fds[::-1]]),
                             y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
                             fill='toself', fillcolor='rgba(204,0,0,0.07)',
                             line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
    fig.add_trace(go.Scatter(x=fds, y=fc['pessimistic'], mode='lines',
                             name='Pesimistis -16%', line=dict(color='#ccc', dash='dash', width=1.2)))
    fig.add_trace(go.Scatter(x=fds, y=fc['optimistic'], mode='lines',
                             name='Optimistis +18%', line=dict(color='#16a34a', dash='dash', width=1.2)))
    fig.add_trace(go.Scatter(x=fds, y=fc['forecast'], mode='lines',
                             name='Prakiraan (LightGBM)', line=dict(color=clr, width=2)))
    lb = LEBARAN_STR.get(2026)
    if lb:
        fig.add_shape(type='line', x0=lb, x1=lb, y0=0, y1=1, yref='paper',
                      line=dict(color='#16a34a', width=1, dash='dash'))
        fig.add_annotation(x=lb, y=1.06, yref='paper', text='Lebaran', showarrow=False,
                           font=dict(size=7.5, color='#16a34a', family='Inter'), xanchor='center')
    if sel_idx is not None:
        x0 = str(fc['ds'].iloc[sel_idx])
        fig.add_shape(type='line', x0=x0, x1=x0, y0=0, y1=1, yref='paper',
                      line=dict(color='#CC0000', width=1, dash='dot'))
    fig.update_layout(**base_layout(height=240,
        xaxis=dict(tickformat='%b %Y', showgrid=True, gridcolor='#F5F5F2',
                   linecolor='#E8E8E5', tickfont=dict(color='#bbb', family='Inter', size=8.5)),
        yaxis=dict(title='Servis/Minggu', showgrid=True, gridcolor='#F5F5F2',
                   linecolor='#E8E8E5', tickfont=dict(color='#bbb', family='Inter', size=8.5),
                   title_font=dict(size=8.5, color='#ccc'))))
    return fig

def chart_shap_bar(kec):
    sr  = shap_results[kec]
    ms  = np.abs(sr['shap_values']).mean(axis=0)
    lbs = [feat_label(f) for f in sr['feat_cols']]
    df  = pd.DataFrame({'F': lbs, 'V': ms}).sort_values('V')
    fig = go.Figure(go.Bar(x=df['V'], y=df['F'], orientation='h',
                           marker_color='#CC0000', marker_line_width=0, opacity=0.75))
    fig.update_layout(**base_layout(height=280, margin=dict(l=8, r=8, t=6, b=30),
        xaxis=dict(title='|SHAP|', showgrid=True, gridcolor='#F5F5F2',
                   linecolor='#E8E8E5', tickfont=dict(color='#bbb', family='Inter', size=8),
                   title_font=dict(size=8.5, color='#ccc')),
        yaxis=dict(showgrid=False, tickfont=dict(color='#444', family='Inter', size=7.5)),
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
    fig.update_layout(**base_layout(barmode='group', height=250,
        margin=dict(l=8, r=8, t=6, b=130),
        xaxis=dict(tickangle=-35, showgrid=False, tickfont=dict(color='#888', family='Inter', size=7.5)),
        yaxis=dict(title='|SHAP|', showgrid=True, gridcolor='#F5F5F2',
                   tickfont=dict(color='#bbb', family='Inter', size=8),
                   title_font=dict(size=8.5, color='#ccc'))))
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
    fig.update_layout(**base_layout(barmode='group', height=220,
        margin=dict(l=8, r=8, t=6, b=60),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F5F5F2',
                   tickfont=dict(color='#bbb', family='Inter', size=8.5),
                   title_font=dict(size=8.5, color='#ccc')),
        xaxis=dict(showgrid=False, tickfont=dict(color='#555', family='Inter', size=10))))
    return fig

def chart_val():
    fig = go.Figure(go.Bar(
        x=[dist_label(k) for k in df_validation['Kecamatan']],
        y=df_validation['MAPE (%)'],
        marker_color=[KEC_COLOR_MAP[k] for k in df_validation['Kecamatan']],
        marker_line_width=0,
        text=df_validation['MAPE (%)'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside', textfont=dict(color='#555', family='Inter', size=9.5)))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                  line=dict(color='#CC0000', width=1, dash='dash'))
    fig.update_layout(**base_layout(height=200, showlegend=False,
        margin=dict(l=8, r=8, t=6, b=40),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F5F5F2',
                   tickfont=dict(color='#bbb', family='Inter', size=8.5),
                   title_font=dict(size=8.5, color='#ccc')),
        xaxis=dict(showgrid=False, tickfont=dict(color='#555', family='Inter', size=10))))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (pure HTML, fixed)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-sidebar" style="position:fixed;left:0;top:0;bottom:0;z-index:9999">
  <div class="sb-logo">A</div>
  <div class="sb-icon active">📊</div>
  <div class="sb-icon">🧠</div>
  <div class="sb-icon">📋</div>
  <div class="sb-icon">✅</div>
  <div class="sb-spacer"></div>
  <div class="sb-icon">⚙️</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WRAPPER — push content right of sidebar
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-left:56px;display:flex;flex-direction:column;height:100vh;overflow:hidden">', unsafe_allow_html=True)

# TOP BAR
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <h1>AHASS Prakiraan Permintaan</h1>
    <p>Prakiraan Permintaan Servis Mingguan · Top 5 District Jakarta · LightGBM + SHAP XAI</p>
  </div>
  <span class="topbar-badge">📅 Prakiraan 2026–2027</span>
</div>
""", unsafe_allow_html=True)

# CONTROL STRIP
st.markdown('<div style="background:#fff;border-bottom:1px solid #E8E8E5;padding:4px 20px">', unsafe_allow_html=True)
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
        f"<p style='font-size:0.58rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#bbb;margin:0'>Model Terbaik</p>"
        f"<p style='font-size:0.75rem;font-weight:600;color:#1a1a1a;margin:1px 0'>LightGBM · 2022–2024 · Uji 2025</p>"
        f"<p style='font-size:0.72rem;font-weight:700;color:#CC0000;margin:0'>MAPE {dist_label(selected_kec)}: {mape_pct:.1f}%</p>",
        unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
fc_row = fc_data.iloc[sel_week_idx]

tab1, tab2, tab3, tab4 = st.tabs(["  PRAKIRAAN  ", "  PENJELASAN XAI  ", "  PERBANDINGAN MODEL  ", "  VALIDASI 2026  "])

# ═══ TAB 1 ═══════════════════════════════════════════════════════════════════
with tab1:
    # KPI strip
    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi-cell">
        <p class="kpi-lbl">District</p>
        <p class="kpi-val" style="color:#1a1a1a;font-size:1.15rem">{dist_label(selected_kec)}</p>
      </div>
      <div class="kpi-cell">
        <p class="kpi-lbl">Prakiraan Dasar</p>
        <p class="kpi-val" style="color:#CC0000">{fc_row['forecast']:,.0f}</p>
        <p class="kpi-sub">servis / minggu</p>
      </div>
      <div class="kpi-cell">
        <p class="kpi-lbl">Optimistis +18%</p>
        <p class="kpi-val" style="color:#16a34a">{fc_row['optimistic']:,.0f}</p>
      </div>
      <div class="kpi-cell">
        <p class="kpi-lbl">Pesimistis -16%</p>
        <p class="kpi-val" style="color:#9ca3af">{fc_row['pessimistic']:,.0f}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown('<p class="panel-lbl" style="padding:10px 0 1px 0">Prakiraan Permintaan</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-ttl">Grafik Prakiraan 52 Minggu</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_forecast(selected_kec, sel_week_idx),
                        use_container_width=True, config={'displayModeBar': False})

    with col_side:
        st.markdown('<p class="panel-lbl" style="padding:10px 0 4px 0">Semua District</p>', unsafe_allow_html=True)
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
              <div class="dist-nums">
                <p class="dist-avg" style="color:{clr}">{fk['forecast'].mean():,.0f}</p>
                <p class="dist-sum">∑ {fk['forecast'].sum():,.0f}/thn</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ═══ TAB 2 ═══════════════════════════════════════════════════════════════════
with tab2:
    fc_xai = forecast_results[selected_kec].iloc[sel_week_idx]
    sr      = shap_results[selected_kec]
    idx     = min(sel_week_idx, len(sr['shap_values']) - 1)
    sv, xv  = sr['shap_values'][idx], sr['X_test'][idx]
    shap_df = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
    shap_df['abs_shap'] = shap_df['shap'].abs()
    top5_shap = shap_df.sort_values('abs_shap', ascending=False).head(5)
    max_abs   = top5_shap['abs_shap'].max()

    col_a, col_b, col_c = st.columns([1, 1.3, 1.7])

    with col_a:
        st.markdown(f"""
        <p class="panel-lbl" style="padding:10px 0 1px 0">Explainable AI</p>
        <p class="panel-ttl">Mengapa Prakiraan Ini?</p>
        <p style="font-size:0.66rem;color:#aaa;margin:0 0 8px 0">
          {fc_xai['ds'].strftime('%d %b %Y')} · {dist_label(selected_kec)} ·
          <strong style="color:#CC0000">{fc_xai['forecast']:,.0f}</strong> servis/minggu
        </p>
        <p style="font-size:0.58rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#bbb;margin:0 0 6px 0">5 Fitur Paling Berpengaruh</p>
        """, unsafe_allow_html=True)

        for _, row in top5_shap.iterrows():
            pos   = row['shap'] > 0
            clr   = "#CC0000" if pos else "#2563eb"
            bg    = "#FFF0F0" if pos else "#EFF6FF"
            ar    = "▲" if pos else "▼"
            bp    = int((row['abs_shap'] / max_abs) * 100)
            st.markdown(f"""
            <div class="feat-row">
              <div class="feat-pill" style="background:{bg};color:{clr}">{ar}</div>
              <div class="feat-body">
                <div class="feat-top">
                  <span class="feat-nm">{row['feature']}</span>
                  <span class="feat-vl">= {row['value']:.1f}</span>
                </div>
                <div class="feat-desc-txt">{feat_desc(row['feature'])}</div>
                <div class="feat-bar">
                  <div class="feat-fill" style="width:{bp}%;background:{clr};opacity:0.65"></div>
                </div>
              </div>
              <span class="feat-imp" style="color:{clr}">{row['shap']:+.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""<div class="note">
          ▲ Merah = menaikkan · ▼ Biru = menurunkan · Batang = besarnya dampak
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<p class="panel-lbl" style="padding:10px 0 1px 0">Kepentingan Global</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="panel-ttl">{dist_label(selected_kec)}</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_shap_bar(selected_kec), use_container_width=True,
                        config={'displayModeBar': False})

    with col_c:
        st.markdown('<p class="panel-lbl" style="padding:10px 0 1px 0">Semua District</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-ttl">Perbandingan SHAP</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_shap_all(), use_container_width=True,
                        config={'displayModeBar': False})
        st.markdown('<p style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#bbb;margin:4px 0 3px 0">Kamus Fitur</p>', unsafe_allow_html=True)
        st.dataframe(FEAT_TABLE, use_container_width=True, hide_index=True, height=140)

# ═══ TAB 3 ═══════════════════════════════════════════════════════════════════
with tab3:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<p class="panel-lbl" style="padding:10px 0 1px 0">Evaluasi Model</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-ttl">MAPE per Model dan District — Periode Uji 2025</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_comp(), use_container_width=True, config={'displayModeBar': False})

        df_cd = df_comparison.copy()
        df_cd.insert(0, 'District', df_cd['Kecamatan'].apply(dist_label))
        df_cd = df_cd[['Model','District','MAPE (%)','RMSE','MAE','R2']].sort_values(['District','MAPE (%)'])
        st.markdown('<p style="font-size:0.65rem;font-weight:700;color:#bbb;text-transform:uppercase;letter-spacing:1px;margin:4px 0 3px 0">Tabel Hasil Lengkap</p>', unsafe_allow_html=True)
        st.dataframe(df_cd, use_container_width=True, height=190, hide_index=True)

    with col_r:
        st.markdown('<p class="panel-lbl" style="padding:10px 0 1px 0">Rata-rata Kinerja</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-ttl">Semua District</p>', unsafe_allow_html=True)
        avg_rows = []
        for mn in df_comparison['Model'].unique():
            sub = df_comparison[df_comparison['Model'] == mn]
            avg_rows.append({'Model': mn,
                             'Avg MAPE (%)': round(sub['MAPE (%)'].mean(), 1),
                             'Avg RMSE': int(sub['RMSE'].mean()),
                             'Avg MAE':  int(sub['MAE'].mean()),
                             'Avg R²':   round(sub['R2'].mean(), 3)})
        df_avg = pd.DataFrame(avg_rows).sort_values('Avg MAPE (%)')
        st.dataframe(df_avg.style.highlight_min(subset=['Avg MAPE (%)'], color='#fff0f0'),
                     use_container_width=True, height=160, hide_index=True)

        st.markdown('<p class="panel-lbl" style="margin:10px 0 4px 0">Interpretasi</p>', unsafe_allow_html=True)
        st.success("**LightGBM** secara konsisten lebih unggul di mayoritas district.")
        st.markdown('<p style="font-size:0.65rem;color:#bbb;margin:4px 0 0 0">Garis merah putus-putus = target MAPE 15%</p>', unsafe_allow_html=True)

# ═══ TAB 4 ═══════════════════════════════════════════════════════════════════
with tab4:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<p class="panel-lbl" style="padding:10px 0 1px 0">Validasi Luar Sampel</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-ttl">Validasi 2026 (Januari – Maret)</p>', unsafe_allow_html=True)
        st.caption("Vs data aktual 2026. Minggu hilang (9–23 Feb) diisi mean. Lebaran dilaporkan terpisah.")
        st.plotly_chart(chart_val(), use_container_width=True, config={'displayModeBar': False})

        dv = df_validation.copy()
        dv.insert(0, 'District', dv['Kecamatan'].apply(dist_label))
        dv = dv[['District','Model','MAPE (%)','RMSE','MAE','Bias','Weeks']].rename(columns={'Weeks':'Minggu'})
        st.dataframe(dv.style.highlight_min(subset=['MAPE (%)'], color='#fff0f0'),
                     use_container_width=True, height=190, hide_index=True)

    with col_r:
        st.markdown('<p class="panel-lbl" style="padding:10px 0 4px 0">Interpretasi Hasil</p>', unsafe_allow_html=True)
        items = [
            ("#16a34a", "#F0FDF4", "✅ Performa Baik",
             "District D dan E mencapai MAPE di bawah 12% selama periode permintaan normal."),
            ("#d97706", "#FFFBEB", "⚠️ Galat Lebih Tinggi",
             "District A dan C menunjukkan galat lebih tinggi akibat volatilitas permintaan yang melebihi pola historis."),
            ("#CC0000", "#FFF5F5", "🔴 Efek Lebaran",
             "MAPE 29–75% selama minggu Lebaran. Penurunan permintaan hari raya di luar rentang pelatihan."),
        ]
        for clr, bg, title, desc in items:
            st.markdown(f"""
            <div class="interp-card" style="background:{bg};border-color:{clr}">
              <p class="interp-title" style="color:{clr}">{title}</p>
              <p class="interp-body">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
