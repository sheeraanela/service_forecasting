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

/* ── App background white ── */
.stApp { background-color: #FFFFFF !important; }

/* ── Kill top padding — target every known Streamlit wrapper ── */
.stApp > div,
.main,
.main > div,
.block-container,
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.block-container {
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Hide all Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
[data-testid="stStatusWidget"]   { display: none !important; }
.stDeployButton                  { display: none !important; }

/* ── RED HEADER ── */
.main-header {
    background-color: #CC0000;
    padding: 2.2rem 3rem;
}
.main-header h1 {
    font-size: 2.4rem; font-weight: 700; color: #FFFFFF !important;
    letter-spacing: 1.5px; margin: 0 0 0.4rem 0; line-height: 1.2;
    text-transform: uppercase;
}
.main-header p { color: rgba(255,255,255,0.78) !important; font-size: 0.88rem; margin: 0; }

/* ── CONTROLS ROW ── */
.ctrl-row {
    display: flex; align-items: flex-start; gap: 2rem;
    padding: 1.2rem 3rem 0.8rem 3rem;
    border-bottom: 1px solid #EEEEEE;
}
.ctrl-model { margin-left: auto; flex-shrink: 0; text-align: left; }
.ctrl-model p { margin: 0; line-height: 1.6; }

/* ── Selectbox ── */
div[data-testid="stSelectbox"] label {
    font-size: 0.68rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    color: #555 !important; margin-bottom: 2px !important;
}
div[data-testid="stSelectbox"] > div > div {
    border-radius: 4px !important; border: 1px solid #CCCCCC !important;
    background: #FFFFFF !important; font-size: 0.88rem !important;
    color: #0D0D0D !important; min-height: 38px !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF; border-bottom: 2px solid #CC0000;
    gap: 0; padding: 0 3rem;
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
.stTabs [data-baseweb="tab-panel"] { padding: 0 3rem !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── SECTION HEADER ── */
.sec-head {
    font-size: 0.9rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #0D0D0D !important;
    border-bottom: 2px solid #CC0000;
    padding-bottom: 0.4rem; margin: 1.8rem 0 1.2rem 0;
}

/* ── METRICS — large, clean ── */
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important; font-weight: 700 !important;
    color: #0D0D0D !important; line-height: 1.15 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important; font-weight: 700 !important;
    color: #666 !important; text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 0.3rem !important;
}
[data-testid="metric-container"] {
    padding: 0.5rem 0 0.5rem 0 !important;
    border: none !important; background: none !important;
}
/* Remove delta arrow if any */
[data-testid="stMetricDelta"] { display: none !important; }

/* ── District mini card ── */
.dist-card { border-top: 3px solid; padding: 10px 12px 8px 0; }
.dist-card-lbl  { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #999; margin: 0; }
.dist-card-val  { font-size: 1.4rem; font-weight: 700; margin: 4px 0 0 0; }
.dist-card-sub  { font-size: 0.65rem; color: #999; margin: 1px 0 0 0; }
.dist-card-sum  { font-size: 0.7rem; font-weight: 600; color: #16a34a; margin: 5px 0 0 0; }
.dist-card-mape { font-size: 0.65rem; color: #999; margin: 3px 0 0 0; }

/* ── XAI note ── */
.xai-note {
    background: #F8F8F8; border-left: 3px solid #CC0000;
    padding: 0.75rem 1rem; font-size: 0.82rem; color: #444; margin-top: 1rem;
}

/* ── Feature row ── */
.feat-row { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.feat-pill {
    width: 22px; height: 22px; border-radius: 5px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
}
.feat-body { flex: 1; min-width: 0; }
.feat-nm   { font-size: 0.82rem; font-weight: 700; color: #0D0D0D; }
.feat-vl   { font-size: 0.68rem; color: #999; margin-left: 5px; }
.feat-desc { font-size: 0.68rem; color: #777; }
.feat-bar  { height: 5px; background: #EEEEEE; border-radius: 3px; margin-top: 5px; }
.feat-fill { height: 5px; border-radius: 3px; }
.feat-imp  { font-size: 0.78rem; font-weight: 700; width: 52px; text-align: right; flex-shrink: 0; }

/* ── Columns: no extra padding ── */
[data-testid="column"] { padding: 0 10px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 4px; }

/* Remove any stray vertical gap between stacked elements */
[data-testid="stVerticalBlock"] { gap: 0 !important; }
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
    dfs_smooth = pd.read_csv(f"{base}/dfs_smooth.csv")
    dfs_smooth['Tanggal Servis'] = pd.to_datetime(dfs_smooth['Tanggal Servis'])
    for kec in forecast_results:
        forecast_results[kec]['ds'] = pd.to_datetime(forecast_results[kec]['ds'])
    return lgb_models, forecast_results, shap_results, config, dfs_smooth

lgb_models, forecast_results, shap_results, config, dfs_smooth = load_assets()

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
    fig.update_layout(**base_layout(height=440, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title='Rata-rata |SHAP|', showgrid=True, gridcolor='#F5F5F5',
                   linecolor='#DDDDDD', tickfont=dict(color='#888', family='DM Sans', size=9),
                   title_font=dict(size=11, color='#999')),
        yaxis=dict(showgrid=False, tickfont=dict(color='#333', family='DM Sans', size=8.5)),
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
    fig.update_layout(**base_layout(barmode='group', height=380,
        margin=dict(l=10, r=10, t=10, b=200),
        xaxis=dict(tickangle=-38, showgrid=False,
                   tickfont=dict(color='#555', family='DM Sans', size=9)),
        yaxis=dict(title='|SHAP|', showgrid=True, gridcolor='#F5F5F5',
                   tickfont=dict(color='#888', family='DM Sans', size=10),
                   title_font=dict(size=11, color='#999'))))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# HEADER — flush to top via CSS, no wrapper div needed
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>AHASS Prakiraan Permintaan</h1>
  <p>Prakiraan Permintaan Servis Mingguan — Top 5 District Jakarta &nbsp;·&nbsp; LightGBM + SHAP XAI</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONTROLS — 3 native Streamlit columns, padded via tab-panel CSS
# ══════════════════════════════════════════════════════════════════════════════
# We use a single-tab trick to get the 3rem side padding consistently
_ctrl, = st.tabs([""])   # invisible single tab for padding context
with _ctrl:
    cc1, cc2, cc3 = st.columns([2, 4, 2])
    with cc1:
        selected_kec = st.selectbox("District", options=TOP5, format_func=dist_label, index=0)
    with cc2:
        fc_data = forecast_results[selected_kec]
        week_options = [f"Minggu {i+1} — {d.strftime('%d %b %Y')}"
                        for i, d in enumerate(fc_data['ds'])]
        sel_week_lbl = st.selectbox("Minggu Prakiraan", options=week_options, index=0)
        sel_week_idx = week_options.index(sel_week_lbl)
    with cc3:
        mape_pct = lgb_models[selected_kec]['mape'] * 100
        st.markdown(
            f"<p style='margin:0;font-size:0.9rem'><strong>Model Terbaik:</strong> LightGBM</p>"
            f"<p style='margin:3px 0;font-size:0.82rem;color:#555'>"
            f"<strong>Pelatihan:</strong> 2022–2024 &nbsp;·&nbsp; <strong>Uji:</strong> 2025</p>"
            f"<p style='margin:0;font-size:0.82rem'><strong>{dist_label(selected_kec)} Test MAPE:</strong> "
            f"<span style='color:#CC0000;font-weight:700'>{mape_pct:.1f}%</span></p>",
            unsafe_allow_html=True)

# ── hide the invisible control tab bar ──
st.markdown("""
<style>
/* Hide the control tab list (first tab group) but keep its panel padding */
div[data-testid="stTabs"]:first-of-type [data-baseweb="tab-list"] {
    display: none !important;
}
div[data-testid="stTabs"]:first-of-type [data-baseweb="tab-panel"] {
    padding-top: 1.2rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #EEEEEE;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
fc_row = fc_data.iloc[sel_week_idx]
tab1, tab2 = st.tabs(["PRAKIRAAN", "PENJELASAN XAI"])

# ═══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    # ── KPI — 4 columns always on one row ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("District",        dist_label(selected_kec))
    with k2: st.metric("Prakiraan Dasar", f"{fc_row['forecast']:,.0f}",  help="servis / minggu")
    with k3: st.metric("Optimistis +18%", f"{fc_row['optimistic']:,.0f}")
    with k4: st.metric("Pesimistis -16%", f"{fc_row['pessimistic']:,.0f}")

    # ── Forecast chart ──
    st.markdown('<div class="sec-head">Grafik Prakiraan 52 Minggu</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_forecast(selected_kec, sel_week_idx),
                    use_container_width=True, config={'displayModeBar': False})

    # ── All-district overview — always 5 columns ──
    st.markdown('<div class="sec-head">Ringkasan Semua District</div>', unsafe_allow_html=True)
    d1, d2, d3, d4, d5 = st.columns(5)
    for col, kec, clr in zip([d1, d2, d3, d4, d5], TOP5, KEC_COLORS):
        fk   = forecast_results[kec]
        mape = lgb_models[kec]['mape'] * 100
        with col:
            st.markdown(f"""
            <div class="dist-card" style="border-color:{clr}">
              <p class="dist-card-lbl">{dist_label(kec)}</p>
              <p class="dist-card-val" style="color:{clr}">{fk['forecast'].mean():,.0f}</p>
              <p class="dist-card-sub">rata-rata/minggu</p>
              <p class="dist-card-sum">∑ {fk['forecast'].sum():,.0f}/thn</p>
              <p class="dist-card-mape">MAPE Uji: {mape:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

# ═══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-head">Explainable AI — Mengapa Prakiraan Ini?</div>',
                unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem;color:#555;margin:-0.6rem 0 1.2rem 0'>"
                "SHAP mengidentifikasi fitur mana yang mendorong prediksi ini dan seberapa besar pengaruhnya.</p>",
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
            f"<p style='font-size:0.95rem;font-weight:700;color:#0D0D0D;margin:0'>"
            f"Prakiraan: <span style='color:#CC0000'>{fc_xai['forecast']:,.0f}</span> servis/minggu</p>"
            f"<p style='font-size:0.78rem;color:#999;margin:2px 0 1.2rem 0'>"
            f"{fc_xai['ds'].strftime('%d %b %Y')} &nbsp;·&nbsp; {dist_label(selected_kec)}</p>",
            unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:1px;color:#555;margin:0 0 12px 0'>5 Fitur Paling Berpengaruh</p>",
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
        st.markdown(f"<p style='font-size:0.82rem;font-weight:700;color:#0D0D0D;margin:0 0 6px 0'>"
                    f"Kepentingan Fitur Global — {dist_label(selected_kec)}</p>",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_shap_bar(selected_kec), use_container_width=True,
                        config={'displayModeBar': False})

    st.markdown('<div class="sec-head">Kepentingan SHAP — Semua District</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_shap_all(), use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="sec-head">Kamus Fitur — Penjelasan Lengkap</div>', unsafe_allow_html=True)
    st.dataframe(FEAT_TABLE, use_container_width=True, hide_index=True, height=530)
