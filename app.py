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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4 {
    font-family: 'DM Sans', sans-serif !important;
    color: #0D0D0D;
}

/* ── Background & hard reset ── */
.stApp { background-color: #FFFFFF; }

/* Kill every possible source of top gap Streamlit injects */
.stApp > div { padding-top: 0 !important; margin-top: 0 !important; }
.main { padding: 0 !important; margin: 0 !important; }
.main > div { padding: 0 !important; margin: 0 !important; }
.main .block-container {
    padding: 0 !important; margin: 0 !important;
    max-width: 100% !important; min-width: 100% !important;
}
.block-container > div { padding: 0 !important; margin: 0 !important; }
section[data-testid="stMain"] { padding: 0 !important; margin: 0 !important; }
[data-testid="stAppViewBlockContainer"] { padding: 0 !important; margin: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
/* The first child markdown that contains the header should have no margin */
[data-testid="stVerticalBlock"] > [data-testid="element-container"]:first-child {
    margin: 0 !important; padding: 0 !important;
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }
[data-testid="stDecoration"]      { display: none !important; }
[data-testid="stStatusWidget"]    { display: none !important; }
.stDeployButton                   { display: none !important; }

/* ── RED HEADER — flush to top ── */
.main-header {
    background-color: #CC0000;
    padding: 2.2rem 3rem;
    margin: 0;
}
.main-header h1 {
    font-size: 2.4rem; font-weight: 700; color: #FFFFFF !important;
    letter-spacing: 1.5px; margin: 0 0 0.5rem 0; line-height: 1.2;
    text-transform: uppercase;
}
.main-header p { color: rgba(255,255,255,0.75) !important; font-size: 0.88rem; margin: 0; }

/* ── CONTENT padding ── */
.content-start { display: none; }

/* Style the Streamlit horizontal block (columns) that holds controls */
[data-testid="stHorizontalBlock"]:first-of-type {
    background: #F8F8F8;
    border: 1px solid #E5E5E5;
    border-left: 4px solid #CC0000;
    padding: 0.8rem 1.5rem;
    margin: 1rem 3rem 1rem 3rem;
    gap: 2rem;
}

/* All content below header gets side padding */
section[data-testid="stMain"] > div > [data-testid="stVerticalBlock"] > [data-testid="element-container"]:not(:first-child),
section[data-testid="stMain"] > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"],
section[data-testid="stMain"] > div > [data-testid="stVerticalBlock"] > [data-testid="stTabs"] {
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* CONTROL BAR — kept for any remaining uses */
.ctrl-bar {
    background: #F8F8F8;
    border: 1px solid #E5E5E5;
    border-left: 4px solid #CC0000;
    padding: 1.1rem 1.5rem;
    margin: 0.8rem 0 1.2rem 0;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] label {
    font-size: 0.72rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    color: #555 !important;
}
div[data-testid="stSelectbox"] > div > div {
    border-radius: 4px !important; border-color: #CCCCCC !important;
    background: #FFFFFF !important; font-size: 0.88rem !important;
    color: #0D0D0D !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF; border-bottom: 2px solid #CC0000;
    gap: 0; padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: #FFFFFF; color: #0D0D0D !important;
    font-weight: 600; font-size: 0.82rem; letter-spacing: 0.5px;
    border: 1.5px solid #0D0D0D; border-bottom: none;
    padding: 0.5rem 1.3rem; border-radius: 0;
}
.stTabs [aria-selected="true"] {
    background: #CC0000 !important; color: #FFFFFF !important;
    border-color: #CC0000 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── SECTION HEADER ── */
.sec-head {
    font-size: 1rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #0D0D0D !important;
    border-bottom: 2px solid #CC0000;
    padding-bottom: 0.4rem; margin: 2rem 0 1rem 0;
}

/* ── METRICS ── */
[data-testid="stMetricValue"] {
    font-size: 2.6rem !important; font-weight: 700 !important; color: #0D0D0D !important;
    line-height: 1.15 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; font-weight: 700 !important; color: #555 !important;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem !important;
}
[data-testid="metric-container"] { padding: 0.5rem 0 !important; }

/* ── XAI note ── */
.xai-note {
    background: #F8F8F8; border-left: 3px solid #CC0000;
    padding: 0.75rem 1rem; font-size: 0.82rem; color: #444; margin-top: 1rem;
}

/* ── Feature row ── */
.feat-row { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.feat-pill {
    width: 24px; height: 24px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700;
}
.feat-body { flex: 1; min-width: 0; }
.feat-nm   { font-size: 0.82rem; font-weight: 700; color: #0D0D0D; }
.feat-vl   { font-size: 0.68rem; color: #999; margin-left: 5px; }
.feat-desc { font-size: 0.68rem; color: #777; }
.feat-bar  { height: 5px; background: #EEEEEE; border-radius: 3px; margin-top: 4px; }
.feat-fill { height: 5px; border-radius: 3px; }
.feat-imp  { font-size: 0.74rem; font-weight: 700; width: 48px; text-align: right; flex-shrink: 0; }

/* ── District summary row ── */
.dist-row {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 0; border-bottom: 1px solid #F0F0F0;
}
.dist-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.dist-name { font-size: 0.82rem; font-weight: 600; color: #0D0D0D; }
.dist-mape { font-size: 0.68rem; color: #999; margin-left: 5px; }
.dist-right { margin-left: auto; text-align: right; }
.dist-avg  { font-size: 0.88rem; font-weight: 700; margin: 0; }
.dist-sum  { font-size: 0.65rem; color: #999; margin: 0; }

/* ── Interp card ── */
.interp-card {
    border-left: 3px solid; padding: 10px 14px; margin-bottom: 10px;
}
.interp-ttl  { font-size: 0.84rem; font-weight: 700; margin: 0 0 3px 0; }
.interp-body { font-size: 0.76rem; color: #555; margin: 0; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 4px; overflow: hidden; }

/* ── Column padding ── */
[data-testid="column"] { padding: 0 6px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

/* ── Content area horizontal padding ── */
.content-pad { padding: 0 3rem; }
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

LEGEND = dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5,
              bgcolor='rgba(255,255,255,0.97)', bordercolor='#DDDDDD', borderwidth=1,
              font=dict(size=10, color='#0D0D0D', family='DM Sans'))

def base_layout(**kw):
    d = dict(
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
        font=dict(color='#0D0D0D', family='DM Sans'),
        xaxis=dict(showgrid=True, gridcolor='#F5F5F5', linecolor='#DDDDDD',
                   tickfont=dict(color='#888', family='DM Sans', size=11)),
        yaxis=dict(showgrid=True, gridcolor='#F5F5F5', linecolor='#DDDDDD',
                   tickfont=dict(color='#888', family='DM Sans', size=11)),
        margin=dict(l=10, r=10, t=10, b=100),
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
                         name='Historis 2025', marker_color='#D6E8FA', marker_line_width=0, opacity=0.85))
    fig.add_trace(go.Scatter(x=pd.concat([fds, fds[::-1]]),
                             y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
                             fill='toself', fillcolor='rgba(204,0,0,0.07)',
                             line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
    fig.add_trace(go.Scatter(x=fds, y=fc['pessimistic'], mode='lines',
                             name='Pesimistis -16%', line=dict(color='#BBBBBB', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=fds, y=fc['optimistic'], mode='lines',
                             name='Optimistis +18%', line=dict(color='#3B6D11', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=fds, y=fc['forecast'], mode='lines',
                             name='Prakiraan (LightGBM)', line=dict(color=clr, width=2.5)))
    lb = LEBARAN_STR.get(2026)
    if lb:
        fig.add_shape(type='line', x0=lb, x1=lb, y0=0, y1=1, yref='paper',
                      line=dict(color='#00AA00', width=1.5, dash='dash'))
        fig.add_annotation(x=lb, y=1.05, yref='paper', text='Lebaran', showarrow=False,
                           font=dict(size=9, color='#00AA00', family='DM Sans'), xanchor='center')
    if sel_idx is not None:
        x0 = str(fc['ds'].iloc[sel_idx])
        fig.add_shape(type='line', x0=x0, x1=x0, y0=0, y1=1, yref='paper',
                      line=dict(color='#CC0000', width=1.5, dash='dot'))
    fig.update_layout(**base_layout(height=420,
        xaxis=dict(tickformat='%b %Y', showgrid=True, gridcolor='#F5F5F5',
                   linecolor='#DDDDDD', tickfont=dict(color='#888', family='DM Sans', size=11)),
        yaxis=dict(title='Jumlah Servis / Minggu', showgrid=True, gridcolor='#F5F5F5',
                   linecolor='#DDDDDD', tickfont=dict(color='#888', family='DM Sans', size=11),
                   title_font=dict(size=11, color='#999'))))
    return fig

def chart_shap_bar(kec):
    sr  = shap_results[kec]
    ms  = np.abs(sr['shap_values']).mean(axis=0)
    lbs = [feat_label(f) for f in sr['feat_cols']]
    df  = pd.DataFrame({'F': lbs, 'V': ms}).sort_values('V')
    fig = go.Figure(go.Bar(x=df['V'], y=df['F'], orientation='h',
                           marker_color='#CC0000', marker_line_width=0, opacity=0.8))
    fig.update_layout(**base_layout(height=430, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title='Rata-rata |SHAP|', showgrid=True, gridcolor='#F5F5F5',
                   linecolor='#DDDDDD', tickfont=dict(color='#888', family='DM Sans', size=10),
                   title_font=dict(size=11, color='#999')),
        yaxis=dict(showgrid=False, tickfont=dict(color='#333', family='DM Sans', size=9)),
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
    fig.update_layout(**base_layout(barmode='group', height=360,
        margin=dict(l=10, r=10, t=10, b=190),
        xaxis=dict(tickangle=-38, showgrid=False, tickfont=dict(color='#666', family='DM Sans', size=9)),
        yaxis=dict(title='|SHAP|', showgrid=True, gridcolor='#F5F5F5',
                   tickfont=dict(color='#888', family='DM Sans', size=10),
                   title_font=dict(size=11, color='#999'))))
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
                       font=dict(size=9, color='#CC0000', family='DM Sans'),
                       xanchor='right', yanchor='bottom')
    fig.update_layout(**base_layout(barmode='group', height=360,
        margin=dict(l=10, r=10, t=10, b=70),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F5F5F5',
                   tickfont=dict(color='#888', family='DM Sans', size=11),
                   title_font=dict(size=11, color='#999')),
        xaxis=dict(showgrid=False, tickfont=dict(color='#333', family='DM Sans', size=12))))
    return fig

def chart_val():
    fig = go.Figure(go.Bar(
        x=[dist_label(k) for k in df_validation['Kecamatan']],
        y=df_validation['MAPE (%)'],
        marker_color=[KEC_COLOR_MAP[k] for k in df_validation['Kecamatan']],
        marker_line_width=0,
        text=df_validation['MAPE (%)'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside', textfont=dict(color='#333', family='DM Sans', size=11)))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                  line=dict(color='#CC0000', width=1, dash='dash'))
    fig.update_layout(**base_layout(height=300, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=50),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F5F5F5',
                   tickfont=dict(color='#888', family='DM Sans', size=11),
                   title_font=dict(size=11, color='#999')),
        xaxis=dict(showgrid=False, tickfont=dict(color='#333', family='DM Sans', size=12))))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>AHASS Prakiraan Permintaan</h1>
  <p>Prakiraan Permintaan Servis Mingguan — Top 5 District Jakarta &nbsp;·&nbsp; LightGBM + SHAP XAI</p>
</div>
<div class="content-start"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONTROLS — no wrapping HTML div, styled via CSS on the column container
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="content-pad">', unsafe_allow_html=True)
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
        f"<p style='margin:0'><strong>Model Terbaik:</strong> LightGBM</p>"
        f"<p style='margin:4px 0;font-size:0.85rem;color:#555'><strong>Pelatihan:</strong> 2022–2024 &nbsp;·&nbsp; <strong>Uji:</strong> 2025</p>"
        f"<p style='margin:0;font-size:0.85rem'><strong>{dist_label(selected_kec)} Test MAPE:</strong> "
        f"<span style='color:#CC0000;font-weight:700'>{mape_pct:.1f}%</span></p>",
        unsafe_allow_html=True)

st.markdown("<hr style='margin:0.8rem 0;border:none;border-top:1px solid #EEEEEE'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
fc_row = fc_data.iloc[sel_week_idx]
tab1, tab2 = st.tabs(["PRAKIRAAN", "PENJELASAN XAI"])

# ═══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    # KPI row — bigger spacing
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("District", dist_label(selected_kec))
    with c2: st.metric("Prakiraan Dasar", f"{fc_row['forecast']:,.0f}", help="servis / minggu")
    with c3: st.metric("Optimistis +18%", f"{fc_row['optimistic']:,.0f}")
    with c4: st.metric("Pesimistis -16%", f"{fc_row['pessimistic']:,.0f}")

    # Forecast chart
    st.markdown('<div class="sec-head">Grafik Prakiraan 52 Minggu</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_forecast(selected_kec, sel_week_idx),
                    use_container_width=True, config={'displayModeBar': False})

    # All-district overview
    st.markdown('<div class="sec-head">Ringkasan Semua District</div>', unsafe_allow_html=True)
    cols = st.columns(len(TOP5))
    for i, (kec, clr) in enumerate(zip(TOP5, KEC_COLORS)):
        fk   = forecast_results[kec]
        mape = lgb_models[kec]['mape'] * 100
        with cols[i]:
            st.markdown(f"""
            <div style="border-top:3px solid {clr};padding:10px 0 4px 0">
              <p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#999;margin:0">{dist_label(kec)}</p>
              <p style="font-size:1.3rem;font-weight:700;color:{clr};margin:4px 0 0 0">{fk['forecast'].mean():,.0f}</p>
              <p style="font-size:0.68rem;color:#999;margin:2px 0 0 0">rata-rata/minggu</p>
              <p style="font-size:0.72rem;color:#16a34a;font-weight:600;margin:4px 0 0 0">∑ {fk['forecast'].sum():,.0f}/thn</p>
              <p style="font-size:0.68rem;color:#999;margin:4px 0 0 0">MAPE Uji: {mape:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

# ═══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-head">Explainable AI — Mengapa Prakiraan Ini?</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem;color:#555;margin:-0.5rem 0 1rem 0'>SHAP mengidentifikasi fitur mana yang mendorong prediksi ini dan seberapa besar pengaruhnya.</p>",
                unsafe_allow_html=True)

    fc_xai  = forecast_results[selected_kec].iloc[sel_week_idx]
    sr      = shap_results[selected_kec]
    idx     = min(sel_week_idx, len(sr['shap_values']) - 1)
    sv, xv  = sr['shap_values'][idx], sr['X_test'][idx]
    shap_df = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
    shap_df['abs_shap'] = shap_df['shap'].abs()
    top5_sh = shap_df.sort_values('abs_shap', ascending=False).head(5)
    max_abs = top5_sh['abs_shap'].max()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(
            f"<p style='font-size:0.95rem;font-weight:700;color:#0D0D0D;margin:0'>Prakiraan: "
            f"<span style='color:#CC0000'>{fc_xai['forecast']:,.0f}</span> servis/minggu</p>"
            f"<p style='font-size:0.78rem;color:#999;margin:2px 0 1rem 0'>"
            f"{fc_xai['ds'].strftime('%d %b %Y')} &nbsp;·&nbsp; {dist_label(selected_kec)}</p>",
            unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#555;margin:0 0 10px 0'>5 Fitur Paling Berpengaruh</p>",
                    unsafe_allow_html=True)

        for _, row in top5_sh.iterrows():
            pos = row['shap'] > 0
            clr = "#CC0000" if pos else "#0066CC"
            bg  = "#FFF0F0" if pos else "#EFF6FF"
            ar  = "▲" if pos else "▼"
            bp  = int((row['abs_shap'] / max_abs) * 100)
            st.markdown(f"""
            <div class="feat-row">
              <div class="feat-pill" style="background:{bg};color:{clr}">{ar}</div>
              <div class="feat-body">
                <div style="display:flex;align-items:baseline;gap:5px">
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

        st.markdown("""<div class="xai-note">
          <strong>Cara membaca:</strong> ▲ Merah = menaikkan prakiraan &nbsp;·&nbsp;
          ▼ Biru = menurunkan prakiraan &nbsp;·&nbsp; Lebar batang = besarnya dampak
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"<p style='font-size:0.82rem;font-weight:700;color:#0D0D0D;margin:0 0 6px 0'>Kepentingan Fitur Global — {dist_label(selected_kec)}</p>",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_shap_bar(selected_kec), use_container_width=True,
                        config={'displayModeBar': False})

    st.markdown('<div class="sec-head">Kepentingan SHAP — Semua District</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_shap_all(), use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="sec-head">Kamus Fitur — Penjelasan Lengkap</div>', unsafe_allow_html=True)
    st.dataframe(FEAT_TABLE, use_container_width=True, hide_index=True, height=530)
