import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="AHASS Prakiraan", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

* { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #fff !important; }

/* hide chrome */
#MainMenu, footer, header, [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stDeployButton,
[data-testid="collapsedControl"], [data-testid="stSidebar"] {
    display: none !important; visibility: hidden !important;
}

/* remove default top padding */
.block-container { padding: 0 !important; max-width: 100% !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; padding: 0 2rem;
    border-bottom: 2px solid #CC0000 !important;
    background: #fff;
}
.stTabs [data-baseweb="tab"] {
    border: 1.5px solid #111; border-bottom: none;
    background: #fff; font-size: 0.8rem; font-weight: 600;
    letter-spacing: 0.5px; padding: 0.45rem 1.2rem; border-radius: 0;
    color: #111 !important;
}
.stTabs [aria-selected="true"] {
    background: #CC0000 !important; color: #fff !important; border-color: #CC0000 !important;
}
[data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding: 0 2rem !important; }

/* metrics */
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 1px; color: #666 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 700 !important; color: #111 !important;
}
[data-testid="metric-container"] { background: none !important; padding: 0 !important; }

/* selectbox */
div[data-testid="stSelectbox"] label {
    font-size: 0.65rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important; color: #666 !important;
}

/* section heading */
.sec { font-size: 0.82rem; font-weight: 700; letter-spacing: 1.2px;
       text-transform: uppercase; border-bottom: 2px solid #CC0000;
       padding-bottom: 0.3rem; margin: 1.6rem 0 1rem 0; color: #111; }

/* feature bar */
.fb { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
.fp { width:20px; height:20px; border-radius:4px; display:flex; align-items:center;
      justify-content:center; font-size:0.65rem; font-weight:700; flex-shrink:0; }
.fn { font-size:0.8rem; font-weight:700; }
.fv { font-size:0.65rem; color:#999; margin-left:4px; }
.fd { font-size:0.65rem; color:#777; }
.fbar { height:4px; background:#eee; border-radius:2px; margin-top:4px; }
.fi { font-size:0.72rem; font-weight:700; width:46px; text-align:right; flex-shrink:0; }

/* note */
.note { background:#f8f8f8; border-left:3px solid #CC0000;
        padding:0.6rem 0.9rem; font-size:0.78rem; color:#444; margin-top:0.8rem; }

/* dist mini */
.dm { border-top:3px solid; padding:8px 0 4px 0; }
.dm-name { font-size:0.6rem; font-weight:700; text-transform:uppercase;
           letter-spacing:0.8px; color:#999; margin:0; }
.dm-val  { font-size:1.3rem; font-weight:700; margin:3px 0 0 0; }
.dm-sub  { font-size:0.62rem; color:#999; margin:1px 0 0 0; }
.dm-sum  { font-size:0.67rem; font-weight:600; color:#16a34a; margin:4px 0 0 0; }
.dm-mape { font-size:0.62rem; color:#999; margin:2px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────────────────────────
FEAT_LABEL = {
    'lag_1':       'Jumlah servis 1 minggu lalu',
    'lag_2':       'Jumlah servis 2 minggu lalu',
    'lag_4':       'Jumlah servis 4 minggu lalu (±1 bln)',
    'lag_8':       'Jumlah servis 8 minggu lalu (±2 bln)',
    'lag_13':      'Jumlah servis 13 minggu lalu (±3 bln)',
    'roll4_mean':  'Rata-rata servis 4 minggu terakhir',
    'roll8_mean':  'Rata-rata servis 8 minggu terakhir',
    'roll13_mean': 'Rata-rata servis 13 minggu terakhir',
    'roll4_std':   'Volatilitas servis 4 minggu terakhir',
    'roll8_std':   'Volatilitas servis 8 minggu terakhir',
    'month':       'Bulan dalam tahun (1–12)',
    'quarter':     'Kuartal dalam tahun (1–4)',
    'year':        'Tahun',
    'month_sin':   'Pola musiman bulan (sinus)',
    'month_cos':   'Pola musiman bulan (kosinus)',
    'is_lebaran':  'Indikator minggu Lebaran',
    'is_holiday':  'Indikator minggu libur nasional',
}
def fdesc(n): return FEAT_LABEL.get(n, n)
def flabel(n): return f"{n} — {fdesc(n)}"
def dlabel(r): return r.replace("KECAMATAN_", "District ")

FEAT_TABLE = pd.DataFrame([{'Nama Fitur': k, 'Penjelasan': v} for k, v in FEAT_LABEL.items()])

# ── load ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    b = "streamlit_assets"
    with open(f"{b}/lgb_models.pkl",      "rb") as f: lgb   = pickle.load(f)
    with open(f"{b}/forecast_results.pkl", "rb") as f: fcr   = pickle.load(f)
    with open(f"{b}/shap_results.pkl",    "rb") as f: shap  = pickle.load(f)
    with open(f"{b}/config.pkl",          "rb") as f: cfg   = pickle.load(f)
    df = pd.read_csv(f"{b}/dfs_smooth.csv")
    df['Tanggal Servis'] = pd.to_datetime(df['Tanggal Servis'])
    for k in fcr: fcr[k]['ds'] = pd.to_datetime(fcr[k]['ds'])
    return lgb, fcr, shap, cfg, df

lgb, fcr, shap_res, cfg, hist_df = load()
TOP5     = cfg['TOP5']
COLORS   = cfg['KEC_COLORS']
CMAP     = dict(zip(TOP5, COLORS))
LEB      = {yr: v[0] for yr, v in cfg['LEBARAN'].items()}

def plotly_base(h=400, **kw):
    d = dict(plot_bgcolor='#fff', paper_bgcolor='#fff',
             font=dict(family='DM Sans', color='#111'),
             margin=dict(l=10, r=10, t=10, b=90),
             xaxis=dict(showgrid=True, gridcolor='#f5f5f5', linecolor='#ddd',
                        tickfont=dict(size=10, color='#888')),
             yaxis=dict(showgrid=True, gridcolor='#f5f5f5', linecolor='#ddd',
                        tickfont=dict(size=10, color='#888')),
             legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center',
                         bgcolor='rgba(255,255,255,0.95)', bordercolor='#ddd',
                         borderwidth=1, font=dict(size=9, color='#555')),
             height=h)
    d.update(kw)
    return d

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#CC0000;padding:2rem 2rem 1.8rem 2rem;margin:0">
  <h1 style="font-size:2.2rem;font-weight:700;color:#fff;letter-spacing:1px;
             text-transform:uppercase;margin:0 0 0.3rem 0">AHASS Prakiraan Permintaan</h1>
  <p style="color:rgba(255,255,255,0.75);font-size:0.85rem;margin:0">
    Prakiraan Permintaan Servis Mingguan — Top 5 District Jakarta · LightGBM + SHAP XAI
  </p>
</div>
""", unsafe_allow_html=True)

# ── CONTROLS ──────────────────────────────────────────────────────────────────
st.markdown("<div style='padding:1rem 2rem 0.8rem 2rem;border-bottom:1px solid #eee'>",
            unsafe_allow_html=True)
c1, c2, c3 = st.columns([2, 4, 2])
with c1:
    sel_kec = st.selectbox("District", TOP5, format_func=dlabel)
with c2:
    fc      = fcr[sel_kec]
    wopts   = [f"Minggu {i+1} — {d.strftime('%d %b %Y')}" for i, d in enumerate(fc['ds'])]
    sel_wk  = st.selectbox("Minggu Prakiraan", wopts)
    wk_idx  = wopts.index(sel_wk)
with c3:
    mape = lgb[sel_kec]['mape'] * 100
    st.markdown(
        f"<p style='margin:0;font-size:0.88rem'><strong>Model Terbaik:</strong> LightGBM</p>"
        f"<p style='margin:3px 0;font-size:0.8rem;color:#666'>"
        f"Pelatihan: 2022–2024 · Uji: 2025</p>"
        f"<p style='margin:0;font-size:0.8rem'>"
        f"<strong>{dlabel(sel_kec)} Test MAPE: "
        f"<span style='color:#CC0000'>{mape:.1f}%</span></strong></p>",
        unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
row = fc.iloc[wk_idx]
tab1, tab2 = st.tabs(["PRAKIRAAN", "PENJELASAN XAI"])

# ═══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # KPI — 4 cols
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("District",        dlabel(sel_kec))
    with k2: st.metric("Prakiraan Dasar", f"{row['forecast']:,.0f}")
    with k3: st.metric("Optimistis +18%", f"{row['optimistic']:,.0f}")
    with k4: st.metric("Pesimistis -16%", f"{row['pessimistic']:,.0f}")

    # Forecast chart
    st.markdown('<div class="sec">Grafik Prakiraan 52 Minggu</div>', unsafe_allow_html=True)
    h   = hist_df[hist_df['Kecamatan Bengkel'] == sel_kec].tail(26)
    fds = fc['ds'].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=h['Tanggal Servis'].astype(str), y=h['Jumlah Servis'],
                         name='Historis 2025', marker_color='#D6E8FA', marker_line_width=0))
    fig.add_trace(go.Scatter(
        x=pd.concat([fds, fds[::-1]]),
        y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
        fill='toself', fillcolor='rgba(204,0,0,0.07)',
        line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
    fig.add_trace(go.Scatter(x=fds, y=fc['pessimistic'], mode='lines',
                             name='Pesimistis -16%',
                             line=dict(color='#BBB', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=fds, y=fc['optimistic'], mode='lines',
                             name='Optimistis +18%',
                             line=dict(color='#3B6D11', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=fds, y=fc['forecast'], mode='lines',
                             name='Prakiraan (LightGBM)',
                             line=dict(color=CMAP[sel_kec], width=2.5)))
    lb = LEB.get(2026)
    if lb:
        fig.add_shape(type='line', x0=lb, x1=lb, y0=0, y1=1, yref='paper',
                      line=dict(color='#00AA00', width=1.2, dash='dash'))
        fig.add_annotation(x=lb, y=1.04, yref='paper', text='Lebaran',
                           showarrow=False, font=dict(size=8, color='#00AA00'), xanchor='center')
    fig.add_shape(type='line',
                  x0=str(fc['ds'].iloc[wk_idx]), x1=str(fc['ds'].iloc[wk_idx]),
                  y0=0, y1=1, yref='paper',
                  line=dict(color='#CC0000', width=1.2, dash='dot'))
    fig.update_layout(**plotly_base(420,
        xaxis=dict(tickformat='%b %Y', showgrid=True, gridcolor='#f5f5f5',
                   linecolor='#ddd', tickfont=dict(size=10, color='#888')),
        yaxis=dict(title='Servis / Minggu', showgrid=True, gridcolor='#f5f5f5',
                   linecolor='#ddd', tickfont=dict(size=10, color='#888'),
                   title_font=dict(size=10, color='#aaa'))))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # District overview — 5 cols
    st.markdown('<div class="sec">Ringkasan Semua District</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, kec, clr in zip(cols, TOP5, COLORS):
        fk = fcr[kec]
        with col:
            st.markdown(f"""
            <div class="dm" style="border-color:{clr}">
              <p class="dm-name">{dlabel(kec)}</p>
              <p class="dm-val" style="color:{clr}">{fk['forecast'].mean():,.0f}</p>
              <p class="dm-sub">rata-rata / minggu</p>
              <p class="dm-sum">∑ {fk['forecast'].sum():,.0f} / thn</p>
              <p class="dm-mape">MAPE Uji: {lgb[kec]['mape']*100:.1f}%</p>
            </div>""", unsafe_allow_html=True)

# ═══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">Explainable AI — Mengapa Prakiraan Ini?</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.83rem;color:#555;margin:-0.5rem 0 1.2rem 0'>"
        "SHAP mengidentifikasi fitur yang mendorong prediksi dan seberapa besar pengaruhnya.</p>",
        unsafe_allow_html=True)

    fc_xai = fcr[sel_kec].iloc[wk_idx]
    sr     = shap_res[sel_kec]
    i      = min(wk_idx, len(sr['shap_values']) - 1)
    sv, xv = sr['shap_values'][i], sr['X_test'][i]
    sdf    = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
    sdf['abs'] = sdf['shap'].abs()
    top5   = sdf.sort_values('abs', ascending=False).head(5)
    mx     = top5['abs'].max()

    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            f"<p style='font-size:0.9rem;font-weight:700;margin:0'>"
            f"Prakiraan: <span style='color:#CC0000'>{fc_xai['forecast']:,.0f}</span> servis/minggu</p>"
            f"<p style='font-size:0.75rem;color:#999;margin:2px 0 1.2rem 0'>"
            f"{fc_xai['ds'].strftime('%d %b %Y')} · {dlabel(sel_kec)}</p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.65rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:1px;color:#555;margin:0 0 10px 0'>5 Fitur Paling Berpengaruh</p>",
            unsafe_allow_html=True)

        for _, row in top5.iterrows():
            pos  = row['shap'] > 0
            clr  = "#CC0000" if pos else "#0055BB"
            bg   = "#FFF0F0" if pos else "#EEF4FF"
            ar   = "▲" if pos else "▼"
            fill = int((row['abs'] / mx) * 100)
            st.markdown(f"""
            <div class="fb">
              <div class="fp" style="background:{bg};color:{clr}">{ar}</div>
              <div style="flex:1;min-width:0">
                <div style="display:flex;align-items:baseline;gap:4px">
                  <span class="fn">{row['feature']}</span>
                  <span class="fv">= {row['value']:.1f}</span>
                </div>
                <div class="fd">{fdesc(row['feature'])}</div>
                <div class="fbar">
                  <div style="height:4px;border-radius:2px;width:{fill}%;background:{clr};opacity:0.6"></div>
                </div>
              </div>
              <span class="fi" style="color:{clr}">{row['shap']:+.1f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="note">
          <strong>Cara membaca:</strong> ▲ Merah = menaikkan prakiraan ·
          ▼ Biru = menurunkan prakiraan · Lebar batang = besarnya dampak
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"<p style='font-size:0.8rem;font-weight:700;margin:0 0 6px 0'>"
                    f"Kepentingan Fitur Global — {dlabel(sel_kec)}</p>",
                    unsafe_allow_html=True)
        ms  = np.abs(sr['shap_values']).mean(axis=0)
        lbs = [flabel(f) for f in sr['feat_cols']]
        df2 = pd.DataFrame({'F': lbs, 'V': ms}).sort_values('V')
        fig2 = go.Figure(go.Bar(x=df2['V'], y=df2['F'], orientation='h',
                                marker_color='#CC0000', opacity=0.8, marker_line_width=0))
        fig2.update_layout(**plotly_base(440,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(title='Rata-rata |SHAP|', showgrid=True, gridcolor='#f5f5f5',
                       linecolor='#ddd', tickfont=dict(size=9, color='#888'),
                       title_font=dict(size=10, color='#aaa')),
            yaxis=dict(showgrid=False, tickfont=dict(size=8, color='#444')),
            legend=dict(orientation='h', y=-0.05)))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # SHAP all districts
    st.markdown('<div class="sec">Kepentingan SHAP — Semua District</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    for kec, clr in zip(TOP5, COLORS):
        sr2  = shap_res[kec]
        ms2  = np.abs(sr2['shap_values']).mean(axis=0)
        lbs2 = [flabel(f) for f in sr2['feat_cols']]
        fig3.add_trace(go.Bar(name=dlabel(kec), x=lbs2, y=ms2,
                              marker_color=clr, marker_line_width=0))
    fig3.update_layout(**plotly_base(360,
        barmode='group',
        margin=dict(l=10, r=10, t=10, b=200),
        xaxis=dict(tickangle=-38, showgrid=False,
                   tickfont=dict(size=8.5, color='#555')),
        yaxis=dict(title='|SHAP|', showgrid=True, gridcolor='#f5f5f5',
                   tickfont=dict(size=9, color='#888'),
                   title_font=dict(size=10, color='#aaa'))))
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    # Feature table
    st.markdown('<div class="sec">Kamus Fitur — Penjelasan Lengkap</div>', unsafe_allow_html=True)
    st.dataframe(FEAT_TABLE, use_container_width=True, hide_index=True, height=530)
