import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="AHASS Prakiraan", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# CSS — single page, no scroll, everything fits in viewport
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
* { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }

/* ── Kill ALL Streamlit padding & scrolling ── */
html, body { overflow: hidden !important; height: 100%; }
.stApp { background: #fff !important; overflow: hidden !important; height: 100vh !important; }
.block-container { padding: 0 !important; max-width: 100% !important; overflow: hidden !important; }
.main { overflow: hidden !important; }
[data-testid="stAppViewBlockContainer"] { padding: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* ── Hide chrome ── */
#MainMenu, footer, header, [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stDeployButton,
[data-testid="collapsedControl"], [data-testid="stSidebar"] {
    display: none !important;
}

/* ── RED HEADER — compact ── */
.app-header {
    background: #CC0000;
    padding: 0.8rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
}
.app-header h1 {
    font-size: 1.25rem; font-weight: 700; color: #fff !important;
    letter-spacing: 1px; text-transform: uppercase; margin: 0;
}
.app-header p { color: rgba(255,255,255,0.75) !important; font-size: 0.72rem; margin: 0; }

/* ── CONTROLS — single row ── */
.ctrl-bar { padding: 0.6rem 2rem; border-bottom: 1px solid #eee; background: #fff; }

/* ── Selectbox compact ── */
div[data-testid="stSelectbox"] label {
    font-size: 0.6rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    color: #888 !important; margin-bottom: 1px !important;
}
div[data-testid="stSelectbox"] > div > div {
    min-height: 30px !important; font-size: 0.82rem !important;
    border-color: #ddd !important; border-radius: 4px !important;
    padding-top: 2px !important; padding-bottom: 2px !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; padding: 0 2rem;
    border-bottom: 2px solid #CC0000 !important;
    background: #fff; flex-shrink: 0;
}
.stTabs [data-baseweb="tab"] {
    border: 1.5px solid #111; border-bottom: none;
    background: #fff; font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.4px; padding: 0.38rem 1.1rem; border-radius: 0;
    color: #111 !important;
}
.stTabs [aria-selected="true"] {
    background: #CC0000 !important; color: #fff !important;
    border-color: #CC0000 !important;
}
[data-baseweb="tab-border"] { display: none !important; }

/* ── Tab panel — scrollable only inside ── */
.stTabs [data-baseweb="tab-panel"] {
    padding: 0.8rem 2rem 0 2rem !important;
    overflow-y: auto !important;
    /* height will be set by JS below */
}

/* ── Metrics compact ── */
[data-testid="stMetricLabel"] {
    font-size: 0.58rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 1px; color: #888 !important;
    margin-bottom: 0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important; font-weight: 700 !important; color: #111 !important;
    line-height: 1.2 !important;
}
[data-testid="metric-container"] { background: none !important; padding: 0 !important; }
[data-testid="stMetricDelta"] { display: none !important; }

/* ── Section heading ── */
.sh {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; border-bottom: 1.5px solid #CC0000;
    padding-bottom: 0.25rem; margin: 1rem 0 0.6rem 0; color: #111;
}

/* ── District mini card ── */
.dm { border-top: 3px solid; padding: 6px 0 2px 0; }
.dm p { margin: 0; }
.dm-n { font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #999; }
.dm-v { font-size: 1.15rem; font-weight: 700; }
.dm-s { font-size: 0.58rem; color: #999; }
.dm-t { font-size: 0.6rem; font-weight: 600; color: #16a34a; margin-top: 2px !important; }
.dm-m { font-size: 0.58rem; color: #999; }

/* ── Feature bar ── */
.fb { display:flex; align-items:center; gap:6px; margin-bottom:7px; }
.fp { width:18px; height:18px; border-radius:4px; display:flex; align-items:center;
      justify-content:center; font-size:0.6rem; font-weight:700; flex-shrink:0; }
.fn { font-size:0.75rem; font-weight:700; }
.fv { font-size:0.6rem; color:#999; margin-left:3px; }
.fd { font-size:0.6rem; color:#777; }
.fbar { height:3px; background:#eee; border-radius:2px; margin-top:3px; }
.fi { font-size:0.68rem; font-weight:700; width:42px; text-align:right; flex-shrink:0; }

/* ── Note box ── */
.note { background:#f8f8f8; border-left:3px solid #CC0000;
        padding:0.5rem 0.8rem; font-size:0.72rem; color:#444; margin-top:0.6rem; }

/* ── Interp card ── */
.ic { border-left:3px solid; padding:7px 12px; margin-bottom:8px; border-radius:0 6px 6px 0; }
.ic-t { font-size:0.75rem; font-weight:700; margin:0 0 2px 0; }
.ic-b { font-size:0.68rem; color:#555; margin:0; }

/* ── Columns ── */
[data-testid="column"] { padding: 0 6px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }
</style>

<script>
// Set tab panel height to fill remaining viewport after header+controls+tabbar
function setHeight() {
    var used = 0;
    ['app-header','ctrl-bar'].forEach(function(cls) {
        var el = document.querySelector('.' + cls);
        if (el) used += el.offsetHeight;
    });
    // Tab bar ~36px, extra padding 4px
    used += 40;
    var panels = document.querySelectorAll('[data-baseweb="tab-panel"]');
    panels.forEach(function(p) { p.style.height = (window.innerHeight - used) + 'px'; });
}
window.addEventListener('load', function() { setTimeout(setHeight, 500); });
window.addEventListener('resize', setHeight);
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS & DATA
# ─────────────────────────────────────────────────────────────────────────────
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

@st.cache_resource
def load():
    b = "streamlit_assets"
    with open(f"{b}/lgb_models.pkl",       "rb") as f: lgb  = pickle.load(f)
    with open(f"{b}/forecast_results.pkl",  "rb") as f: fcr  = pickle.load(f)
    with open(f"{b}/shap_results.pkl",     "rb") as f: shap = pickle.load(f)
    with open(f"{b}/config.pkl",           "rb") as f: cfg  = pickle.load(f)
    df  = pd.read_csv(f"{b}/dfs_smooth.csv")
    df['Tanggal Servis'] = pd.to_datetime(df['Tanggal Servis'])
    dv  = pd.read_csv(f"{b}/validation_metrics.csv")
    for k in fcr: fcr[k]['ds'] = pd.to_datetime(fcr[k]['ds'])
    return lgb, fcr, shap, cfg, df, dv

lgb, fcr, shap_res, cfg, hist_df, df_val_raw = load()
TOP5  = cfg['TOP5']
COLS  = cfg['KEC_COLORS']
CMAP  = dict(zip(TOP5, COLS))
LEB   = {yr: v[0] for yr, v in cfg['LEBARAN'].items()}
df_val = df_val_raw.copy()
df_val['District'] = df_val['Kecamatan'].apply(dlabel)
val_mape = dict(zip(df_val['Kecamatan'], df_val['MAPE (%)']))

def get_actuals(kec):
    fc  = shap_res[kec]['feat_cols']
    xt  = shap_res[kec]['X_test']
    li  = fc.index('lag_1')
    return np.array([xt[i+1, li] if i+1 < len(xt) else np.nan for i in range(13)])

def pbase(h=260, **kw):
    d = dict(
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        font=dict(family='DM Sans', color='#111'),
        margin=dict(l=8, r=8, t=8, b=70),
        height=h,
        xaxis=dict(showgrid=True, gridcolor='#f5f5f5', linecolor='#ddd',
                   tickfont=dict(size=9, color='#888')),
        yaxis=dict(showgrid=True, gridcolor='#f5f5f5', linecolor='#ddd',
                   tickfont=dict(size=9, color='#888')),
        legend=dict(orientation='h', y=-0.28, x=0.5, xanchor='center',
                    font=dict(size=8.5, color='#555'),
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#ddd', borderwidth=1))
    d.update(kw)
    return d

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <h1>AHASS Prakiraan Permintaan</h1>
    <p>Prakiraan Servis Mingguan — Top 5 District Jakarta · LightGBM + SHAP XAI</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="ctrl-bar">', unsafe_allow_html=True)
cc1, cc2, cc3 = st.columns([2, 4, 3])
with cc1:
    sel_kec = st.selectbox("District", TOP5, format_func=dlabel)
with cc2:
    fc = fcr[sel_kec]
    wopts = [f"Minggu {i+1} — {d.strftime('%d %b %Y')}" for i, d in enumerate(fc['ds'])]
    sel_wk = st.selectbox("Minggu Prakiraan", wopts)
    wk_idx = wopts.index(sel_wk)
with cc3:
    mape = lgb[sel_kec]['mape'] * 100
    st.markdown(
        f"<span style='font-size:0.82rem'><strong>Model:</strong> LightGBM &nbsp;·&nbsp; "
        f"<strong>Pelatihan:</strong> 2022–2024 &nbsp;·&nbsp; <strong>Uji:</strong> 2025 &nbsp;·&nbsp; "
        f"<strong>MAPE {dlabel(sel_kec)}:</strong> "
        f"<span style='color:#CC0000;font-weight:700'>{mape:.1f}%</span></span>",
        unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
row    = fc.iloc[wk_idx]
tab1, tab2, tab3 = st.tabs(["  PRAKIRAAN  ", "  PENJELASAN XAI  ", "  VALIDASI 2026  "])

# ══ TAB 1 — PRAKIRAAN ════════════════════════════════════════════════════════
with tab1:
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("District",        dlabel(sel_kec))
    with k2: st.metric("Prakiraan Dasar", f"{row['forecast']:,.0f}")
    with k3: st.metric("Optimistis +18%", f"{row['optimistic']:,.0f}")
    with k4: st.metric("Pesimistis -16%", f"{row['pessimistic']:,.0f}")

    # Main chart + district summary side by side
    ch, sd = st.columns([3, 1])

    with ch:
        st.markdown('<div class="sh">Grafik Prakiraan 52 Minggu</div>', unsafe_allow_html=True)
        hist = hist_df[hist_df['Kecamatan Bengkel'] == sel_kec].tail(26)
        fds  = fc['ds'].astype(str)
        fig  = go.Figure()
        fig.add_trace(go.Bar(x=hist['Tanggal Servis'].astype(str), y=hist['Jumlah Servis'],
                             name='Historis 2025', marker_color='#D6E8FA', marker_line_width=0))
        fig.add_trace(go.Scatter(
            x=pd.concat([fds, fds[::-1]]),
            y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
            fill='toself', fillcolor='rgba(204,0,0,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
        fig.add_trace(go.Scatter(x=fds, y=fc['pessimistic'], mode='lines',
                                 name='Pesimistis', line=dict(color='#BBB', dash='dash', width=1.2)))
        fig.add_trace(go.Scatter(x=fds, y=fc['optimistic'], mode='lines',
                                 name='Optimistis', line=dict(color='#3B6D11', dash='dash', width=1.2)))
        fig.add_trace(go.Scatter(x=fds, y=fc['forecast'], mode='lines',
                                 name='Prakiraan (LightGBM)',
                                 line=dict(color=CMAP[sel_kec], width=2.2)))
        lb = LEB.get(2026)
        if lb:
            fig.add_shape(type='line', x0=lb, x1=lb, y0=0, y1=1, yref='paper',
                          line=dict(color='#00AA00', width=1, dash='dash'))
            fig.add_annotation(x=lb, y=1.05, yref='paper', text='Lebaran',
                               showarrow=False, font=dict(size=8, color='#00AA00'), xanchor='center')
        fig.add_shape(type='line',
                      x0=str(fc['ds'].iloc[wk_idx]), x1=str(fc['ds'].iloc[wk_idx]),
                      y0=0, y1=1, yref='paper',
                      line=dict(color='#CC0000', width=1, dash='dot'))
        fig.update_layout(**pbase(310,
            margin=dict(l=8, r=8, t=8, b=80),
            xaxis=dict(tickformat='%b %Y', showgrid=True, gridcolor='#f5f5f5',
                       linecolor='#ddd', tickfont=dict(size=9, color='#888')),
            yaxis=dict(title='Servis/Minggu', showgrid=True, gridcolor='#f5f5f5',
                       linecolor='#ddd', tickfont=dict(size=9, color='#888'),
                       title_font=dict(size=9, color='#bbb'))))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with sd:
        st.markdown('<div class="sh">Semua District</div>', unsafe_allow_html=True)
        for kec, clr in zip(TOP5, COLS):
            fk = fcr[kec]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #f5f5f5">
              <div style="width:8px;height:8px;border-radius:50%;background:{clr};flex-shrink:0"></div>
              <div style="flex:1">
                <span style="font-size:0.75rem;font-weight:600;color:#111">{dlabel(kec)}</span>
                <span style="font-size:0.62rem;color:#bbb;margin-left:4px">MAPE {lgb[kec]['mape']*100:.1f}%</span>
              </div>
              <div style="text-align:right">
                <div style="font-size:0.82rem;font-weight:700;color:{clr}">{fk['forecast'].mean():,.0f}</div>
                <div style="font-size:0.6rem;color:#bbb">∑ {fk['forecast'].sum():,.0f}/thn</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ══ TAB 2 — PENJELASAN XAI ═══════════════════════════════════════════════════
with tab2:
    fc_xai = fcr[sel_kec].iloc[wk_idx]
    sr     = shap_res[sel_kec]
    ii     = min(wk_idx, len(sr['shap_values']) - 1)
    sv, xv = sr['shap_values'][ii], sr['X_test'][ii]
    sdf    = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
    sdf['abs'] = sdf['shap'].abs()
    top5   = sdf.sort_values('abs', ascending=False).head(5)
    mx     = top5['abs'].max()

    col_a, col_b, col_c = st.columns([1, 1.3, 1.8])

    with col_a:
        st.markdown('<div class="sh">5 Fitur Teratas</div>', unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:0.8rem;font-weight:700;margin:0'>"
            f"Prakiraan: <span style='color:#CC0000'>{fc_xai['forecast']:,.0f}</span> servis/minggu</p>"
            f"<p style='font-size:0.68rem;color:#999;margin:1px 0 10px 0'>"
            f"{fc_xai['ds'].strftime('%d %b %Y')} · {dlabel(sel_kec)}</p>",
            unsafe_allow_html=True)
        for _, r in top5.iterrows():
            pos  = r['shap'] > 0
            clr  = "#CC0000" if pos else "#0055BB"
            bg   = "#FFF0F0" if pos else "#EEF4FF"
            fill = int((r['abs'] / mx) * 100)
            st.markdown(f"""
            <div class="fb">
              <div class="fp" style="background:{bg};color:{clr}">{"▲" if pos else "▼"}</div>
              <div style="flex:1;min-width:0">
                <div><span class="fn">{r['feature']}</span><span class="fv">= {r['value']:.1f}</span></div>
                <div class="fd">{fdesc(r['feature'])}</div>
                <div class="fbar"><div style="height:3px;border-radius:2px;width:{fill}%;background:{clr};opacity:0.6"></div></div>
              </div>
              <span class="fi" style="color:{clr}">{r['shap']:+.1f}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="note">
          ▲ Merah = menaikkan · ▼ Biru = menurunkan · Lebar = dampak
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<div class="sh">Kepentingan Fitur — {dlabel(sel_kec)}</div>', unsafe_allow_html=True)
        ms  = np.abs(sr['shap_values']).mean(axis=0)
        lbs = [flabel(f) for f in sr['feat_cols']]
        df2 = pd.DataFrame({'F': lbs, 'V': ms}).sort_values('V')
        fig2 = go.Figure(go.Bar(x=df2['V'], y=df2['F'], orientation='h',
                                marker_color='#CC0000', opacity=0.8, marker_line_width=0))
        fig2.update_layout(**pbase(360,
            margin=dict(l=8, r=8, t=8, b=20),
            xaxis=dict(title='Rata-rata |SHAP|', showgrid=True, gridcolor='#f5f5f5',
                       linecolor='#ddd', tickfont=dict(size=8, color='#888'),
                       title_font=dict(size=9, color='#bbb')),
            yaxis=dict(showgrid=False, tickfont=dict(size=7.5, color='#444')),
            legend=dict(orientation='h', y=-0.05)))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with col_c:
        st.markdown('<div class="sh">Perbandingan SHAP Semua District</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for kec, clr in zip(TOP5, COLS):
            sr2 = shap_res[kec]
            ms2 = np.abs(sr2['shap_values']).mean(axis=0)
            fig3.add_trace(go.Bar(name=dlabel(kec),
                                  x=[flabel(f) for f in sr2['feat_cols']],
                                  y=ms2, marker_color=clr, marker_line_width=0))
        fig3.update_layout(**pbase(220,
            barmode='group',
            margin=dict(l=8, r=8, t=8, b=150),
            xaxis=dict(tickangle=-38, showgrid=False, tickfont=dict(size=7.5, color='#666')),
            yaxis=dict(title='|SHAP|', showgrid=True, gridcolor='#f5f5f5',
                       tickfont=dict(size=8, color='#888'),
                       title_font=dict(size=9, color='#bbb'))))
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="sh" style="margin-top:0.5rem">Kamus Fitur</div>', unsafe_allow_html=True)
        st.dataframe(FEAT_TABLE, use_container_width=True, hide_index=True, height=130)

# ══ TAB 3 — VALIDASI 2026 ════════════════════════════════════════════════════
with tab3:
    avg_mape = df_val['MAPE (%)'].mean()
    best     = df_val.loc[df_val['MAPE (%)'].idxmin()]
    worst    = df_val.loc[df_val['MAPE (%)'].idxmax()]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Rata-rata MAPE",    f"{avg_mape:.1f}%")
    with k2: st.metric("Terbaik",  dlabel(best['Kecamatan']),  delta=f"MAPE {best['MAPE (%)']:.1f}%")
    with k3: st.metric("Terlemah", dlabel(worst['Kecamatan']), delta=f"MAPE {worst['MAPE (%)']:.1f}%", delta_color="inverse")
    with k4: st.metric("Periode",  "Jan – Mar 2026", help="13 minggu")

    # Main: forecast vs actual charts (2 rows × 3 cols condensed) + metrics table
    chart_col, table_col = st.columns([2.5, 1])

    with chart_col:
        st.markdown('<div class="sh">Prakiraan vs Data Aktual 2026 per District</div>', unsafe_allow_html=True)
        # 3 + 2 layout
        row1 = st.columns(3)
        row2_cols = st.columns(3)  # only use first 2

        for i, (kec, clr) in enumerate(zip(TOP5, COLS)):
            col = row1[i] if i < 3 else row2_cols[i - 3]
            fc_kec = fcr[kec]
            hist_k = hist_df[hist_df['Kecamatan Bengkel'] == kec].tail(26)
            fds    = fc_kec['ds'].astype(str)
            act13  = get_actuals(kec)
            ds13   = fc_kec['ds'].iloc[:13].astype(str)
            mv     = val_mape.get(kec, 0)

            fv = go.Figure()
            fv.add_trace(go.Bar(x=hist_k['Tanggal Servis'].astype(str), y=hist_k['Jumlah Servis'],
                                name='Hist 2025', marker_color='#D6E8FA', marker_line_width=0, opacity=0.8))
            fv.add_trace(go.Scatter(
                x=pd.concat([fds, fds[::-1]]),
                y=pd.concat([fc_kec['optimistic'], fc_kec['pessimistic'][::-1]]),
                fill='toself', fillcolor='rgba(204,0,0,0.07)',
                line=dict(color='rgba(0,0,0,0)'), name='Rentang', showlegend=False))
            fv.add_trace(go.Scatter(x=fds, y=fc_kec['forecast'], mode='lines',
                                    name='Prakiraan', line=dict(color=clr, width=1.5)))
            fv.add_trace(go.Scatter(x=ds13, y=act13, mode='lines+markers',
                                    name='Aktual 2026',
                                    line=dict(color='#CC0000', width=2),
                                    marker=dict(size=4, color='#CC0000')))
            lb = LEB.get(2026)
            if lb:
                fv.add_shape(type='line', x0=lb, x1=lb, y0=0, y1=1, yref='paper',
                             line=dict(color='#00AA00', width=1, dash='dash'))
            fv.update_layout(**pbase(205,
                title=dict(text=f"{dlabel(kec)}  MAPE={mv:.1f}%",
                           font=dict(size=10, color='#111'), x=0.5, xanchor='center'),
                margin=dict(l=8, r=8, t=28, b=65),
                xaxis=dict(tickformat='%b %Y', showgrid=True, gridcolor='#f5f5f5',
                           linecolor='#ddd', tickfont=dict(size=7.5, color='#888')),
                yaxis=dict(showgrid=True, gridcolor='#f5f5f5',
                           tickfont=dict(size=8, color='#888')),
                legend=dict(orientation='h', y=-0.38, x=0.5, xanchor='center',
                            font=dict(size=7.5), bgcolor='rgba(255,255,255,0.9)')))
            with col:
                st.plotly_chart(fv, use_container_width=True, config={'displayModeBar': False})

    with table_col:
        st.markdown('<div class="sh">Metrik Validasi</div>', unsafe_allow_html=True)
        disp = df_val[['District','MAPE (%)','RMSE','MAE','Bias']].sort_values('MAPE (%)')
        st.dataframe(
            disp.style
                .highlight_min(subset=['MAPE (%)'], color='#d4edda')
                .highlight_max(subset=['MAPE (%)'], color='#f8d7da')
                .format({'MAPE (%)': '{:.1f}%', 'RMSE': '{:,}', 'MAE': '{:,}', 'Bias': '{:+,}'}),
            use_container_width=True, hide_index=True, height=250)

        st.markdown('<div class="sh" style="margin-top:0.8rem">Interpretasi</div>', unsafe_allow_html=True)
        for clr, bg, title, desc in [
            ("#16a34a","#f0fdf4","✅ Performa Baik",
             "District A, C, E — MAPE < 12%, konsisten dengan uji 2025."),
            ("#d97706","#fffbeb","⚠️ Perlu Perhatian",
             "District B & D melebihi target 15% MAPE."),
            ("#CC0000","#fff5f5","🔴 Efek Lebaran",
             "Penurunan permintaan Lebaran di luar rentang pelatihan."),
        ]:
            st.markdown(f"""
            <div class="ic" style="background:{bg};border-color:{clr}">
              <p class="ic-t" style="color:{clr}">{title}</p>
              <p class="ic-b">{desc}</p>
            </div>""", unsafe_allow_html=True)
