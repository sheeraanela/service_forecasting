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
        color: #1a1a2e;
    }

    /* ── Background ── */
    .stApp { background-color: #f0f2f5 !important; }
    .main .block-container { padding: 0 1.5rem 2rem 1.5rem !important; max-width: 100% !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    section[data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}

    /* ── Top header bar ── */
    .exec-header {
        background: #ffffff;
        padding: 1.2rem 2rem;
        margin: 0 -1.5rem 1.5rem -1.5rem;
        border-bottom: 1px solid #e8eaed;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .exec-header-left { display: flex; align-items: center; gap: 1rem; }
    .exec-logo {
        width: 42px; height: 42px; background: #CC0000;
        border-radius: 10px; display: flex; align-items: center;
        justify-content: center; font-weight: 800; font-size: 1.1rem;
        color: white; flex-shrink: 0;
    }
    .exec-title { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; margin: 0; }
    .exec-subtitle { font-size: 0.78rem; color: #6b7280; margin: 0.1rem 0 0 0; }
    .exec-date {
        background: #f8f9fa; border: 1px solid #e8eaed; border-radius: 8px;
        padding: 0.4rem 0.9rem; font-size: 0.8rem; color: #6b7280; font-weight: 500;
    }

    /* ── Card base ── */
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.8px;
        text-transform: uppercase; color: #9ca3af; margin-bottom: 0.2rem;
    }
    .card-value {
        font-size: 1.9rem; font-weight: 700; color: #CC0000; line-height: 1.2;
    }
    .card-delta-up   { font-size: 0.78rem; font-weight: 600; color: #16a34a; }
    .card-delta-down { font-size: 0.78rem; font-weight: 600; color: #dc2626; }

    /* ── Section header inside card ── */
    .chart-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.4rem 0.5rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .chart-card-title {
        font-size: 0.78rem; font-weight: 600; color: #6b7280;
        letter-spacing: 0.4px; margin-bottom: 0.15rem;
    }
    .chart-card-subtitle {
        font-size: 1rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.8rem;
    }

    /* ── Pill buttons (filter bar) ── */
    .pill-bar { display: flex; gap: 0.5rem; margin-bottom: 1.2rem; }
    .pill {
        padding: 0.35rem 1rem; border-radius: 20px; font-size: 0.8rem;
        font-weight: 600; cursor: pointer; border: 1.5px solid #e8eaed;
        background: #ffffff; color: #6b7280;
    }
    .pill-active { background: #CC0000; color: #ffffff; border-color: #CC0000; }

    /* ── Section label ── */
    .section-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
        text-transform: uppercase; color: #9ca3af;
        margin: 1.5rem 0 0.6rem 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff; border-radius: 12px 12px 0 0;
        border-bottom: 2px solid #f0f2f5; gap: 0; padding: 0 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #9ca3af !important;
        font-weight: 600; font-size: 0.82rem; letter-spacing: 0.3px;
        border: none; border-bottom: 2px solid transparent;
        padding: 0.8rem 1.3rem; border-radius: 0;
        font-family: 'Inter', sans-serif !important;
    }
    .stTabs [aria-selected="true"] {
        color: #CC0000 !important; border-bottom: 2px solid #CC0000 !important;
        background: transparent !important;
    }

    /* ── Metric overrides ── */
    [data-testid="stMetricValue"] { font-family: 'Inter', sans-serif !important; font-weight: 700; color: #CC0000 !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { font-family: 'Inter', sans-serif !important; font-weight: 600; color: #9ca3af !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.8px; }
    [data-testid="metric-container"] {
        background: #ffffff; border-radius: 14px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
    }

    /* ── Selectbox ── */
    div[data-testid="stSelectbox"] label {
        font-weight: 600 !important; font-size: 0.72rem !important;
        text-transform: uppercase !important; letter-spacing: 0.8px !important;
        color: #9ca3af !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 10px !important; border-color: #e8eaed !important;
        background: #ffffff !important;
    }

    /* ── Control bar ── */
    .ctrl-card {
        background: #ffffff; border-radius: 14px;
        padding: 1rem 1.4rem; margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
        border-left: 4px solid #CC0000;
    }

    /* ── XAI note ── */
    .xai-note {
        background: #fff5f5; border-left: 3px solid #CC0000;
        padding: 0.8rem 1rem; font-size: 0.82rem; color: #6b7280;
        margin-top: 1rem; border-radius: 0 8px 8px 0;
    }

    /* ── Info / success / warning / error cards ── */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
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

# ── Load assets ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = "streamlit_assets"
    with open(f"{base}/lgb_models.pkl",      "rb") as f: lgb_models      = pickle.load(f)
    with open(f"{base}/forecast_results.pkl", "rb") as f: forecast_results = pickle.load(f)
    with open(f"{base}/shap_results.pkl",     "rb") as f: shap_results    = pickle.load(f)
    with open(f"{base}/config.pkl",           "rb") as f: config          = pickle.load(f)
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

def dist_label(raw):
    """KECAMATAN_A → District A"""
    return raw.replace("KECAMATAN_", "District ")

LEGEND = dict(
    orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5,
    bgcolor='rgba(255,255,255,0.97)', bordercolor='#e8eaed', borderwidth=1,
    font=dict(size=10, color='#1a1a2e', family='Inter'),
)

def base_layout(**kwargs):
    d = dict(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e', family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='#f5f5f5', linecolor='#e8eaed',
                   tickfont=dict(color='#6b7280', family='Inter', size=11)),
        yaxis=dict(showgrid=True, gridcolor='#f5f5f5', linecolor='#e8eaed',
                   tickfont=dict(color='#6b7280', family='Inter', size=11)),
        margin=dict(l=20, r=20, t=30, b=110),
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
        name='Historis 2025', marker_color='#dbeafe', opacity=0.9,
        marker_line_width=0))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_ds, fc_ds[::-1]]),
        y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
        fill='toself', fillcolor='rgba(204,0,0,0.07)',
        line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
    fig.add_trace(go.Scatter(
        x=fc_ds, y=fc['pessimistic'], mode='lines',
        name='Pesimistis -16%', line=dict(color='#9ca3af', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(
        x=fc_ds, y=fc['optimistic'], mode='lines',
        name='Optimistis +18%', line=dict(color='#16a34a', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(
        x=fc_ds, y=fc['forecast'], mode='lines',
        name='Prakiraan (LightGBM)', line=dict(color=color, width=2.5)))
    lb26 = LEBARAN_STR.get(2026)
    if lb26:
        fig.add_shape(type='line', x0=lb26, x1=lb26, y0=0, y1=1, yref='paper',
                      line=dict(color='#16a34a', width=1.5, dash='dash'))
        fig.add_annotation(x=lb26, y=1.03, yref='paper', text='Lebaran',
                           showarrow=False, font=dict(size=9, color='#16a34a', family='Inter'),
                           xanchor='center')
    if sel_idx is not None:
        fig.add_shape(type='line',
                      x0=str(fc['ds'].iloc[sel_idx]), x1=str(fc['ds'].iloc[sel_idx]),
                      y0=0, y1=1, yref='paper',
                      line=dict(color='#CC0000', width=1.5, dash='dot'))
    fig.update_layout(**base_layout(
        height=400,
        xaxis=dict(tickformat='%b %Y', showgrid=True, gridcolor='#f5f5f5',
                   linecolor='#e8eaed', tickfont=dict(color='#6b7280', family='Inter', size=11)),
        yaxis=dict(title='Jumlah Servis / Minggu', showgrid=True, gridcolor='#f5f5f5',
                   linecolor='#e8eaed', tickfont=dict(color='#6b7280', family='Inter', size=11),
                   title_font=dict(size=11, color='#9ca3af')),
    ))
    return fig

def make_shap_bar(kec):
    sr        = shap_results[kec]
    mean_shap = np.abs(sr['shap_values']).mean(axis=0)
    labels    = [feat_label(f) for f in sr['feat_cols']]
    df        = pd.DataFrame({'Fitur': labels, 'Kepentingan': mean_shap}).sort_values('Kepentingan')
    fig = go.Figure(go.Bar(
        x=df['Kepentingan'], y=df['Fitur'], orientation='h',
        marker_color='#CC0000', marker_line_width=0,
        marker=dict(color='#CC0000', opacity=0.85)))
    fig.update_layout(**base_layout(
        height=500, margin=dict(l=20, r=20, t=10, b=40),
        xaxis=dict(title='Rata-rata |Nilai SHAP|', showgrid=True, gridcolor='#f5f5f5',
                   linecolor='#e8eaed', tickfont=dict(color='#6b7280', family='Inter', size=10),
                   title_font=dict(size=11, color='#9ca3af')),
        yaxis=dict(showgrid=False, linecolor='#e8eaed',
                   tickfont=dict(color='#1a1a2e', family='Inter', size=9)),
        legend=dict(orientation='h', y=-0.05),
    ))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="exec-header">
  <div class="exec-header-left">
    <div class="exec-logo">A</div>
    <div>
      <p class="exec-title">AHASS Prakiraan Permintaan</p>
      <p class="exec-subtitle">Prakiraan Permintaan Servis Mingguan · Top 5 District Jakarta · LightGBM + SHAP XAI</p>
    </div>
  </div>
  <div class="exec-date">📅 &nbsp;Prakiraan 2026 – 2027</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONTROL BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ctrl-card">', unsafe_allow_html=True)
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 3, 2])
with col_ctrl1:
    selected_kec = st.selectbox("District", options=TOP5,
                                format_func=dist_label, index=0)
with col_ctrl2:
    fc_data      = forecast_results[selected_kec]
    week_options = [f"Minggu {i+1} — {d.strftime('%d %b %Y')}" for i, d in enumerate(fc_data['ds'])]
    selected_week_label = st.selectbox("Minggu Prakiraan", options=week_options, index=0)
    selected_week_idx   = week_options.index(selected_week_label)
with col_ctrl3:
    st.markdown(
        f"<p style='font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#9ca3af;margin:0'>Model Terbaik</p>"
        f"<p style='font-size:0.95rem;font-weight:700;color:#1a1a2e;margin:0.1rem 0'>LightGBM</p>"
        f"<p style='font-size:0.75rem;color:#6b7280;margin:0.1rem 0'>Pelatihan: 2022–2024 &nbsp;·&nbsp; Uji: 2025</p>"
        f"<p style='font-size:0.75rem;color:#CC0000;font-weight:600;margin:0'>MAPE Uji {dist_label(selected_kec)}: {lgb_models[selected_kec]['mape']*100:.1f}%</p>",
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["  PRAKIRAAN  ", "  PENJELASAN XAI  ", "  PERBANDINGAN MODEL  ", "  VALIDASI 2026  "])

# ═══ TAB 1 ═══════════════════════════════════════════════════════════════════
with tab1:
    fc_row = fc_data.iloc[selected_week_idx]

    # ── KPI metric cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("District",        dist_label(selected_kec))
    with c2: st.metric("Prakiraan Dasar", f"{fc_row['forecast']:,.0f}", help="servis / minggu")
    with c3: st.metric("Optimistis +18%", f"{fc_row['optimistic']:,.0f}")
    with c4: st.metric("Pesimistis -16%", f"{fc_row['pessimistic']:,.0f}")

    # ── Forecast chart ──
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-title">Prakiraan Permintaan</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-subtitle">Grafik Prakiraan 52 Minggu</p>', unsafe_allow_html=True)
    st.plotly_chart(make_forecast_chart(selected_kec, selected_week_idx), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── All-district overview ──
    st.markdown('<p class="section-label">Ringkasan Semua District</p>', unsafe_allow_html=True)
    cols = st.columns(len(TOP5))
    for i, (kec, color) in enumerate(zip(TOP5, KEC_COLORS)):
        fc_kec = forecast_results[kec]
        with cols[i]:
            st.markdown(f"""
            <div style="background:#ffffff;border-radius:14px;padding:1rem 1.2rem;
                        box-shadow:0 1px 3px rgba(0,0,0,0.07),0 4px 16px rgba(0,0,0,0.04);
                        border-top:3px solid {color};">
                <p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.8px;color:#9ca3af;margin:0">{dist_label(kec)}</p>
                <p style="font-size:1.4rem;font-weight:700;color:{color};margin:0.3rem 0 0 0">
                    {fc_kec['forecast'].mean():,.0f}
                </p>
                <p style="font-size:0.72rem;color:#6b7280;margin:0.1rem 0">
                    rata-rata / minggu
                </p>
                <p style="font-size:0.72rem;color:#16a34a;font-weight:600;margin:0.2rem 0 0 0">
                    ▲ tahunan {fc_kec['forecast'].sum():,.0f}
                </p>
                <p style="font-size:0.7rem;color:#9ca3af;margin:0.3rem 0 0 0">
                    MAPE Uji = {lgb_models[kec]['mape']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

# ═══ TAB 2 ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-title">Explainable AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-subtitle">Mengapa Prakiraan Ini?</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.82rem;color:#6b7280;margin-bottom:1rem'>"
        "SHAP (SHapley Additive exPlanations) mengidentifikasi fitur mana yang mendorong "
        "prediksi ini dan seberapa besar pengaruhnya.</p>",
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        fc_row_xai = forecast_results[selected_kec].iloc[selected_week_idx]
        st.markdown(
            f"<p style='font-size:1rem;font-weight:700;color:#CC0000;margin:0'>"
            f"{fc_row_xai['forecast']:,.0f} <span style='font-size:0.75rem;font-weight:500;"
            f"color:#6b7280'>servis/minggu</span></p>"
            f"<p style='font-size:0.75rem;color:#9ca3af;margin:0.1rem 0 1rem 0'>"
            f"{fc_row_xai['ds'].strftime('%d %b %Y')} &nbsp;·&nbsp; {dist_label(selected_kec)}</p>",
            unsafe_allow_html=True
        )

        sr      = shap_results[selected_kec]
        idx     = min(selected_week_idx, len(sr['shap_values']) - 1)
        sv      = sr['shap_values'][idx]
        xv      = sr['X_test'][idx]
        shap_df = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
        shap_df['abs_shap'] = shap_df['shap'].abs()
        shap_df = shap_df.sort_values('abs_shap', ascending=False).head(5)
        max_abs = shap_df['abs_shap'].max()

        st.markdown(
            "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.8px;color:#9ca3af;margin-bottom:0.6rem'>5 Fitur Paling Berpengaruh</p>",
            unsafe_allow_html=True
        )
        for _, row in shap_df.iterrows():
            direction = "▲" if row['shap'] > 0 else "▼"
            clr       = "#CC0000" if row['shap'] > 0 else "#2563eb"
            bar_w     = int((row['abs_shap'] / max_abs) * 100)
            ca, cb, cc = st.columns([1, 3, 4])
            with ca:
                st.markdown(
                    f"<div style='background:{clr}15;border-radius:8px;padding:0.4rem 0.5rem;"
                    f"text-align:center;margin-top:0.3rem'>"
                    f"<span style='color:{clr};font-size:0.9rem;font-weight:700'>{direction}</span></div>",
                    unsafe_allow_html=True)
            with cb:
                st.markdown(
                    f"<p style='font-size:0.82rem;font-weight:700;color:#1a1a2e;margin:0.3rem 0 0 0'>{row['feature']}</p>"
                    f"<p style='font-size:0.72rem;color:#6b7280;margin:0.1rem 0'>{feat_desc(row['feature'])}</p>"
                    f"<p style='font-size:0.7rem;color:#9ca3af;margin:0'>nilai = {row['value']:.1f}</p>",
                    unsafe_allow_html=True)
            with cc:
                st.progress(bar_w)
                st.markdown(
                    f"<p style='font-size:0.72rem;color:{clr};font-weight:600;margin:0.1rem 0 0.6rem 0'>"
                    f"dampak: {row['shap']:+.1f}</p>",
                    unsafe_allow_html=True)

        st.markdown("""<div class="xai-note">
            <strong>Cara membaca:</strong> ▲ Merah = fitur <em>menaikkan</em> prakiraan &nbsp;·&nbsp;
            ▼ Biru = fitur <em>menurunkan</em> prakiraan &nbsp;·&nbsp; Lebar batang = besarnya dampak
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(
            f"<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.8px;color:#9ca3af;margin-bottom:0.3rem'>"
            f"Kepentingan Fitur Global — {dist_label(selected_kec)}</p>",
            unsafe_allow_html=True)
        st.plotly_chart(make_shap_bar(selected_kec), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SHAP all districts ──
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-title">Perbandingan SHAP</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-subtitle">Kepentingan SHAP — Semua District</p>', unsafe_allow_html=True)
    fig_all = go.Figure()
    for kec, color in zip(TOP5, KEC_COLORS):
        sr        = shap_results[kec]
        mean_shap = np.abs(sr['shap_values']).mean(axis=0)
        labels    = [feat_label(f) for f in sr['feat_cols']]
        fig_all.add_trace(go.Bar(name=dist_label(kec), x=labels, y=mean_shap,
                                 marker_color=color, marker_line_width=0))
    fig_all.update_layout(**base_layout(
        barmode='group', height=420,
        margin=dict(l=20, r=20, t=10, b=220),
        xaxis=dict(tickangle=-40, showgrid=False, linecolor='#e8eaed',
                   tickfont=dict(color='#6b7280', family='Inter', size=9)),
        yaxis=dict(title='Rata-rata |Nilai SHAP|', showgrid=True, gridcolor='#f5f5f5',
                   linecolor='#e8eaed', tickfont=dict(color='#6b7280', family='Inter'),
                   title_font=dict(size=11, color='#9ca3af')),
    ))
    st.plotly_chart(fig_all, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature dictionary table ──
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-title">Referensi</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-subtitle">Kamus Fitur — Penjelasan Lengkap</p>', unsafe_allow_html=True)
    st.dataframe(FEAT_TABLE, use_container_width=True, hide_index=True, height=530)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══ TAB 3 ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-title">Evaluasi Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-subtitle">Perbandingan Model — Periode Uji 2025</p>', unsafe_allow_html=True)

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

    st.markdown(
        "<p style='font-size:0.78rem;font-weight:600;color:#6b7280;margin-bottom:0.4rem'>"
        "Rata-rata Kinerja di Semua District</p>",
        unsafe_allow_html=True)
    st.dataframe(df_avg.style.highlight_min(subset=['Rata-rata MAPE (%)'], color='#fff5f5'),
                 use_container_width=True, height=180)

    st.markdown(
        "<p style='font-size:0.78rem;font-weight:600;color:#6b7280;margin:1.2rem 0 0.4rem 0'>"
        "MAPE per Model dan District</p>",
        unsafe_allow_html=True)
    model_colors = {'XGBoost': '#E6A817', 'LightGBM': '#CC0000', 'Stacking': '#639922'}
    fig_comp = go.Figure()
    for model_name in df_comparison['Model'].unique():
        sub = df_comparison[df_comparison['Model'] == model_name]
        fig_comp.add_trace(go.Bar(
            name=model_name,
            x=[dist_label(k) for k in sub['Kecamatan']],
            y=sub['MAPE (%)'],
            marker_color=model_colors.get(model_name, '#999'),
            marker_line_width=0))
    fig_comp.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                       line=dict(color='#CC0000', width=1, dash='dash'))
    fig_comp.add_annotation(x=1, y=15, xref='paper', text='Target 15%',
                            showarrow=False, font=dict(size=9, color='#CC0000', family='Inter'),
                            xanchor='right', yanchor='bottom')
    fig_comp.update_layout(**base_layout(
        barmode='group', height=380,
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#f5f5f5',
                   linecolor='#e8eaed', tickfont=dict(color='#6b7280', family='Inter'),
                   title_font=dict(size=11, color='#9ca3af')),
        xaxis=dict(showgrid=False, linecolor='#e8eaed',
                   tickfont=dict(color='#1a1a2e', family='Inter', size=11)),
    ))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown(
        "<p style='font-size:0.78rem;font-weight:600;color:#6b7280;margin:1rem 0 0.4rem 0'>"
        "Tabel Hasil Lengkap</p>",
        unsafe_allow_html=True)
    df_comp_disp = df_comparison.copy()
    df_comp_disp['District'] = df_comp_disp['Kecamatan'].apply(dist_label)
    df_comp_disp = df_comp_disp[['Model','District','MAPE (%)','RMSE','MAE','R2']].sort_values(['District','MAPE (%)'])
    st.dataframe(df_comp_disp, use_container_width=True, height=350, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══ TAB 4 ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-title">Evaluasi Luar Sampel</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-card-subtitle">Validasi Luar Sampel 2026</p>', unsafe_allow_html=True)
    st.info("Validasi terhadap data aktual 2026 (Januari – Maret 2026). Minggu yang hilang (9–23 Feb) diisi dengan nilai rata-rata. Minggu Lebaran dilaporkan terpisah karena volatilitas permintaan yang ekstrem.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            "<p style='font-size:0.78rem;font-weight:600;color:#6b7280;margin-bottom:0.4rem'>"
            "Metrik Validasi Keseluruhan</p>",
            unsafe_allow_html=True)
        df_val_disp = df_validation.copy()
        df_val_disp['District'] = df_val_disp['Kecamatan'].apply(dist_label)
        df_val_disp = df_val_disp[['District','Model','MAPE (%)','RMSE','MAE','Bias','Weeks']].rename(columns={'Weeks':'Minggu'})
        st.dataframe(df_val_disp.style.highlight_min(subset=['MAPE (%)'], color='#fff5f5'),
                     use_container_width=True, height=220, hide_index=True)
    with col2:
        st.markdown(
            "<p style='font-size:0.78rem;font-weight:600;color:#6b7280;margin-bottom:0.4rem'>"
            "MAPE per District</p>",
            unsafe_allow_html=True)
        fig_val = go.Figure(go.Bar(
            x=[dist_label(k) for k in df_validation['Kecamatan']],
            y=df_validation['MAPE (%)'],
            marker_color=[KEC_COLOR_MAP[k] for k in df_validation['Kecamatan']],
            marker_line_width=0,
            text=df_validation['MAPE (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            textfont=dict(color='#1a1a2e', family='Inter', size=11),
        ))
        fig_val.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                          line=dict(color='#CC0000', width=1, dash='dash'))
        fig_val.update_layout(**base_layout(
            height=280, showlegend=False,
            margin=dict(l=20, r=20, t=20, b=40),
            yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#f5f5f5',
                       linecolor='#e8eaed', tickfont=dict(color='#6b7280', family='Inter'),
                       title_font=dict(size=11, color='#9ca3af')),
            xaxis=dict(showgrid=False, linecolor='#e8eaed',
                       tickfont=dict(color='#1a1a2e', family='Inter', size=11)),
        ))
        st.plotly_chart(fig_val, use_container_width=True)

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.8px;color:#9ca3af;margin:1.2rem 0 0.6rem 0'>Interpretasi</p>",
        unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("**✅ Performa Baik**\n\nDistrict D dan E mencapai MAPE di bawah 12% selama periode permintaan normal.")
    with c2:
        st.warning("**⚠️ Galat Lebih Tinggi**\n\nDistrict A dan C menunjukkan galat lebih tinggi akibat volatilitas permintaan yang melebihi pola historis.")
    with c3:
        st.error("**🔴 Efek Lebaran**\n\nMAPE 29–75% selama minggu Lebaran. Penurunan permintaan hari raya berada di luar rentang pelatihan model.")
    st.markdown('</div>', unsafe_allow_html=True)
