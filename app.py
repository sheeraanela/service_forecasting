import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import matplotlib; matplotlib.use('Agg')

st.set_page_config(page_title="AHASS Prakiraan", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }

/* Hide chrome only */
#MainMenu, footer, header,
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, [data-testid="collapsedControl"],
[data-testid="stSidebar"] { display: none !important; }

/* Clean background */
.stApp { background: #f8f8f8 !important; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* Red header block */
.app-hdr {
    background: #CC0000; padding: 14px 0 12px 0;
    margin: 0 -2rem 1.2rem -2rem;
}
.app-hdr-title {
    color: #fff !important; font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase; margin: 0;
}
.app-hdr-sub { color: rgba(255,255,255,0.72) !important; font-size: 0.68rem; margin: 2px 0 0 0; }

/* Control strip */
.ctrl-strip {
    background: #fff; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 1rem;
    border: 1px solid #e8e8e8;
}

/* Metrics */
[data-testid="stMetricLabel"] {
    font-size: 0.6rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.9px; color: #888 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important; font-weight: 700 !important; color: #111 !important;
}
[data-testid="metric-container"] {
    background: #fff; border-radius: 8px; padding: 12px 16px !important;
    border: 1px solid #e8e8e8;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* Selectbox */
div[data-testid="stSelectbox"] label {
    font-size: 0.62rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.9px !important; color: #888 !important;
}
div[data-testid="stSelectbox"] > div > div {
    font-size: 0.82rem !important; border-color: #ddd !important;
    border-radius: 6px !important; background: #fff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; padding: 0; border-bottom: 2px solid #CC0000 !important; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border: 1.5px solid #222; border-bottom: none; background: #fff;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.3px;
    padding: 6px 18px; border-radius: 0; color: #222 !important;
}
.stTabs [aria-selected="true"] {
    background: #CC0000 !important; color: #fff !important; border-color: #CC0000 !important;
}
[data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 1rem 0 0 0 !important; }

/* Section heading */
.sec {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #888; border-bottom: 1.5px solid #CC0000;
    padding-bottom: 4px; margin: 0 0 10px 0;
}

/* White card */
.card {
    background: #fff; border-radius: 8px; padding: 14px 16px;
    border: 1px solid #e8e8e8; height: 100%;
}

/* District row */
.drow { display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #f2f2f2; }
.ddot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dname { font-size:0.75rem; font-weight:600; color:#111; }
.dmape { font-size:0.6rem; color:#bbb; margin-left:3px; }
.dval  { font-size:0.78rem; font-weight:700; }
.dsum  { font-size:0.58rem; color:#bbb; }

/* Feature row */
.frow { display:flex; align-items:center; gap:6px; margin-bottom:8px; }
.fpill { width:18px; height:18px; border-radius:4px; flex-shrink:0;
         display:flex; align-items:center; justify-content:center; font-size:0.6rem; font-weight:700; }
.fname { font-size:0.75rem; font-weight:700; color:#111; }
.fval  { font-size:0.6rem; color:#bbb; margin-left:3px; }
.fdesc { font-size:0.6rem; color:#777; }
.fbar  { height:3px; background:#eee; border-radius:2px; margin-top:3px; }
.fimp  { font-size:0.68rem; font-weight:700; width:40px; text-align:right; flex-shrink:0; }

/* Note */
.note { background:#fff5f5; border-left:2px solid #CC0000; padding:6px 10px;
        font-size:0.68rem; color:#555; margin-top:8px; border-radius:0 4px 4px 0; }

/* Interp card */
.icard { border-left:3px solid; padding:8px 12px; margin-bottom:8px; border-radius:0 6px 6px 0; }
.ititle { font-size:0.72rem; font-weight:700; margin:0 0 2px 0; }
.ibody  { font-size:0.65rem; color:#555; margin:0; }

/* Col gaps */
[data-testid="column"] { padding: 0 6px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data & helpers ─────────────────────────────────────────────────────────────
FEAT = {
    'lag_1':'Jumlah servis 1 minggu lalu', 'lag_2':'Jumlah servis 2 minggu lalu',
    'lag_4':'Jumlah servis 4 minggu lalu (±1 bln)', 'lag_8':'Jumlah servis 8 minggu lalu (±2 bln)',
    'lag_13':'Jumlah servis 13 minggu lalu (±3 bln)', 'roll4_mean':'Rata-rata servis 4 minggu terakhir',
    'roll8_mean':'Rata-rata servis 8 minggu terakhir', 'roll13_mean':'Rata-rata servis 13 minggu terakhir',
    'roll4_std':'Volatilitas servis 4 minggu terakhir', 'roll8_std':'Volatilitas servis 8 minggu terakhir',
    'month':'Bulan (1–12)', 'quarter':'Kuartal (1–4)', 'year':'Tahun',
    'month_sin':'Pola musiman (sinus)', 'month_cos':'Pola musiman (kosinus)',
    'is_lebaran':'Indikator minggu Lebaran', 'is_holiday':'Indikator libur nasional',
}
FEAT_DF = pd.DataFrame([{'Nama Fitur':k, 'Penjelasan':v} for k,v in FEAT.items()])
def fd(n): return FEAT.get(n, n)
def fl(n): return f"{n} — {fd(n)}"
def dl(r): return r.replace("KECAMATAN_", "District ")

@st.cache_resource
def load():
    b = "streamlit_assets"
    with open(f"{b}/lgb_models.pkl","rb") as f: lgb=pickle.load(f)
    with open(f"{b}/forecast_results.pkl","rb") as f: fcr=pickle.load(f)
    with open(f"{b}/shap_results.pkl","rb") as f: shp=pickle.load(f)
    with open(f"{b}/config.pkl","rb") as f: cfg=pickle.load(f)
    df = pd.read_csv(f"{b}/dfs_smooth.csv")
    df['Tanggal Servis'] = pd.to_datetime(df['Tanggal Servis'])
    dv = pd.read_csv(f"{b}/validation_metrics.csv")
    for k in fcr: fcr[k]['ds'] = pd.to_datetime(fcr[k]['ds'])
    return lgb, fcr, shp, cfg, df, dv

lgb, fcr, shp, cfg, hdf, dv = load()
TOP5 = cfg['TOP5']; COLS = cfg['KEC_COLORS']; CM = dict(zip(TOP5, COLS))
LEB  = {yr: v[0] for yr, v in cfg['LEBARAN'].items()}
dv   = dv.copy(); dv['District'] = dv['Kecamatan'].apply(dl)
vmape = dict(zip(dv['Kecamatan'], dv['MAPE (%)']))

def get_actuals(kec):
    fc = shp[kec]['feat_cols']; xt = shp[kec]['X_test']; li = fc.index('lag_1')
    return np.array([xt[i+1,li] if i+1<len(xt) else np.nan for i in range(13)])

def chart_base(h=300):
    return dict(
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        height=h, font=dict(family='Inter', size=9, color='#555'),
        margin=dict(l=8, r=8, t=8, b=80),
        xaxis=dict(showgrid=True, gridcolor='#f2f2f2', linecolor='#e0e0e0', zeroline=False,
                   tickfont=dict(size=8, color='#aaa')),
        yaxis=dict(showgrid=True, gridcolor='#f2f2f2', linecolor='#e0e0e0', zeroline=False,
                   tickfont=dict(size=8, color='#aaa')),
        legend=dict(orientation='h', y=-0.28, x=0.5, xanchor='center',
                    font=dict(size=8, color='#666'),
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#eee', borderwidth=1))

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-hdr">
  <div style="padding:0 2rem">
    <p class="app-hdr-title">AHASS Prakiraan Permintaan</p>
    <p class="app-hdr-sub">Prakiraan Servis Mingguan — Top 5 District Jakarta · LightGBM + SHAP XAI</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── CONTROLS ──────────────────────────────────────────────────────────────────
with st.container():
    c1, c2, c3 = st.columns([2, 4, 3])
    with c1:
        sel = st.selectbox("District", TOP5, format_func=dl)
    with c2:
        fc = fcr[sel]
        wo = [f"Minggu {i+1} — {d.strftime('%d %b %Y')}" for i,d in enumerate(fc['ds'])]
        sw = st.selectbox("Minggu Prakiraan", wo)
        wi = wo.index(sw)
    with c3:
        mp = lgb[sel]['mape'] * 100
        st.markdown(
            f"<div style='padding:8px 0'>"
            f"<span style='font-size:0.8rem;color:#444'>"
            f"<b>Model:</b> LightGBM &nbsp;·&nbsp; <b>Pelatihan:</b> 2022–2024 &nbsp;·&nbsp; "
            f"<b>Uji:</b> 2025 &nbsp;·&nbsp; "
            f"<b>MAPE {dl(sel)}:</b> <span style='color:#CC0000;font-weight:700'>{mp:.1f}%</span>"
            f"</span></div>", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
row = fc.iloc[wi]
t1, t2, t3 = st.tabs(["  PRAKIRAAN  ", "  PENJELASAN XAI  ", "  VALIDASI 2026  "])

# ══════════════════════════════════════════════════════════════════════════════
with t1:
    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("District", dl(sel))
    with k2: st.metric("Prakiraan Dasar", f"{row['forecast']:,.0f}")
    with k3: st.metric("Optimistis +18%", f"{row['optimistic']:,.0f}")
    with k4: st.metric("Pesimistis -16%", f"{row['pessimistic']:,.0f}")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    gc, sc = st.columns([3, 1])

    with gc:
        st.markdown('<p class="sec">Grafik Prakiraan 52 Minggu</p>', unsafe_allow_html=True)
        h = hdf[hdf['Kecamatan Bengkel']==sel].tail(26)
        fds = fc['ds'].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=h['Tanggal Servis'].astype(str), y=h['Jumlah Servis'],
                             name='Historis 2025', marker_color='#DAEAF8', marker_line_width=0))
        fig.add_trace(go.Scatter(
            x=pd.concat([fds, fds[::-1]]),
            y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
            fill='toself', fillcolor='rgba(204,0,0,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
        fig.add_trace(go.Scatter(x=fds, y=fc['pessimistic'], mode='lines', name='Pesimistis',
                                 line=dict(color='#CCC', dash='dash', width=1.2)))
        fig.add_trace(go.Scatter(x=fds, y=fc['optimistic'], mode='lines', name='Optimistis',
                                 line=dict(color='#3B6D11', dash='dash', width=1.2)))
        fig.add_trace(go.Scatter(x=fds, y=fc['forecast'], mode='lines', name='Prakiraan (LightGBM)',
                                 line=dict(color=CM[sel], width=2)))
        lb = LEB.get(2026)
        if lb:
            fig.add_shape(type='line', x0=lb, x1=lb, y0=0, y1=1, yref='paper',
                          line=dict(color='#00AA00', width=1, dash='dash'))
            fig.add_annotation(x=lb, y=1.04, yref='paper', text='Lebaran', showarrow=False,
                               font=dict(size=8, color='#00AA00'), xanchor='center')
        fig.add_shape(type='line',
                      x0=str(fc['ds'].iloc[wi]), x1=str(fc['ds'].iloc[wi]),
                      y0=0, y1=1, yref='paper',
                      line=dict(color='#CC0000', width=1, dash='dot'))
        layout = chart_base(300)
        layout['xaxis']['tickformat'] = '%b %Y'
        layout['yaxis']['title'] = 'Servis/Minggu'
        layout['yaxis']['title_font'] = dict(size=9, color='#bbb')
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with sc:
        st.markdown('<p class="sec">Semua District</p>', unsafe_allow_html=True)
        for kec, clr in zip(TOP5, COLS):
            fk = fcr[kec]
            st.markdown(f"""
            <div class="drow">
              <div class="ddot" style="background:{clr}"></div>
              <div style="flex:1">
                <span class="dname">{dl(kec)}</span>
                <span class="dmape">MAPE {lgb[kec]['mape']*100:.1f}%</span>
              </div>
              <div style="text-align:right">
                <div class="dval" style="color:{clr}">{fk['forecast'].mean():,.0f}</div>
                <div class="dsum">∑ {fk['forecast'].sum():,.0f}/thn</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with t2:
    xai = fcr[sel].iloc[wi]
    sr  = shp[sel]
    ii  = min(wi, len(sr['shap_values'])-1)
    sv, xv = sr['shap_values'][ii], sr['X_test'][ii]
    sdf = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
    sdf['abs'] = sdf['shap'].abs()
    top5f = sdf.sort_values('abs', ascending=False).head(5)
    mx    = top5f['abs'].max()

    col_a, col_b, col_c = st.columns([1, 1.4, 1.6])

    with col_a:
        st.markdown('<p class="sec">5 Fitur Paling Berpengaruh</p>', unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:0.8rem;font-weight:700;margin:0 0 2px 0'>"
            f"Prakiraan: <span style='color:#CC0000'>{xai['forecast']:,.0f}</span> servis/minggu</p>"
            f"<p style='font-size:0.65rem;color:#bbb;margin:0 0 12px 0'>"
            f"{xai['ds'].strftime('%d %b %Y')} · {dl(sel)}</p>", unsafe_allow_html=True)

        for _, r in top5f.iterrows():
            pos  = r['shap'] > 0
            clr  = "#CC0000" if pos else "#0055BB"
            bg   = "#FFF0F0" if pos else "#EEF4FF"
            fill = int((r['abs'] / mx) * 100)
            st.markdown(f"""
            <div class="frow">
              <div class="fpill" style="background:{bg};color:{clr}">{"▲" if pos else "▼"}</div>
              <div style="flex:1;min-width:0">
                <div><span class="fname">{r['feature']}</span><span class="fval">= {r['value']:.1f}</span></div>
                <div class="fdesc">{fd(r['feature'])}</div>
                <div class="fbar"><div style="height:3px;border-radius:2px;width:{fill}%;background:{clr};opacity:0.6"></div></div>
              </div>
              <span class="fimp" style="color:{clr}">{r['shap']:+.1f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="note">▲ Merah = menaikkan · ▼ Biru = menurunkan · Lebar = dampak</div>',
                    unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<p class="sec">Kepentingan Fitur — {dl(sel)}</p>', unsafe_allow_html=True)
        ms  = np.abs(sr['shap_values']).mean(axis=0)
        df2 = pd.DataFrame({'F': [fl(f) for f in sr['feat_cols']], 'V': ms}).sort_values('V')
        f2  = go.Figure(go.Bar(x=df2['V'], y=df2['F'], orientation='h',
                               marker_color='#CC0000', opacity=0.75, marker_line_width=0))
        l2  = chart_base(360)
        l2['margin'] = dict(l=8, r=8, t=8, b=20)
        l2['xaxis']['title'] = '|SHAP|'
        l2['xaxis']['title_font'] = dict(size=9, color='#bbb')
        l2['yaxis']['tickfont'] = dict(size=7.5, color='#444')
        l2['yaxis']['showgrid'] = False
        l2['legend'] = dict(orientation='h', y=-0.04)
        f2.update_layout(**l2)
        st.plotly_chart(f2, use_container_width=True, config={'displayModeBar': False})

    with col_c:
        st.markdown('<p class="sec">SHAP Semua District</p>', unsafe_allow_html=True)
        f3 = go.Figure()
        for kec, clr in zip(TOP5, COLS):
            sr2 = shp[kec]
            ms2 = np.abs(sr2['shap_values']).mean(axis=0)
            f3.add_trace(go.Bar(name=dl(kec),
                                x=[fl(f) for f in sr2['feat_cols']],
                                y=ms2, marker_color=clr, marker_line_width=0))
        l3 = chart_base(210)
        l3['barmode'] = 'group'
        l3['margin']  = dict(l=8, r=8, t=8, b=130)
        l3['xaxis']['tickangle'] = -35
        l3['xaxis']['showgrid']  = False
        l3['xaxis']['tickfont']  = dict(size=7.5, color='#888')
        l3['yaxis']['title']     = '|SHAP|'
        l3['yaxis']['title_font']= dict(size=9, color='#bbb')
        f3.update_layout(**l3)
        st.plotly_chart(f3, use_container_width=True, config={'displayModeBar': False})

        st.markdown('<p class="sec">Kamus Fitur</p>', unsafe_allow_html=True)
        st.dataframe(FEAT_DF, use_container_width=True, hide_index=True, height=145)

# ══════════════════════════════════════════════════════════════════════════════
with t3:
    am   = dv['MAPE (%)'].mean()
    best = dv.loc[dv['MAPE (%)'].idxmin()]
    wrst = dv.loc[dv['MAPE (%)'].idxmax()]

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("Rata-rata MAPE", f"{am:.1f}%")
    with k2: st.metric("Terbaik",  dl(best['Kecamatan']), delta=f"MAPE {best['MAPE (%)']:.1f}%")
    with k3: st.metric("Terlemah", dl(wrst['Kecamatan']), delta=f"MAPE {wrst['MAPE (%)']:.1f}%", delta_color="inverse")
    with k4: st.metric("Periode", "Jan – Mar 2026")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    lc, rc = st.columns([2.4, 1])

    with lc:
        st.markdown('<p class="sec">Prakiraan vs Aktual 2026 per District</p>', unsafe_allow_html=True)
        r1 = st.columns(3)
        r2 = st.columns(3)
        for i, (kec, clr) in enumerate(zip(TOP5, COLS)):
            col = r1[i] if i < 3 else r2[i-3]
            fck = fcr[kec]
            hk  = hdf[hdf['Kecamatan Bengkel']==kec].tail(26)
            fds = fck['ds'].astype(str)
            act = get_actuals(kec)
            d13 = fck['ds'].iloc[:13].astype(str)
            mv  = vmape.get(kec, 0)

            fv = go.Figure()
            fv.add_trace(go.Bar(x=hk['Tanggal Servis'].astype(str), y=hk['Jumlah Servis'],
                                name='Hist 2025', marker_color='#DAEAF8', marker_line_width=0, opacity=0.8))
            fv.add_trace(go.Scatter(
                x=pd.concat([fds, fds[::-1]]),
                y=pd.concat([fck['optimistic'], fck['pessimistic'][::-1]]),
                fill='toself', fillcolor='rgba(204,0,0,0.05)',
                line=dict(color='rgba(0,0,0,0)'), name='Rentang', showlegend=False))
            fv.add_trace(go.Scatter(x=fds, y=fck['forecast'], mode='lines', name='Prakiraan',
                                    line=dict(color=clr, width=1.5)))
            fv.add_trace(go.Scatter(x=d13, y=act, mode='lines+markers', name='Aktual 2026',
                                    line=dict(color='#CC0000', width=1.8),
                                    marker=dict(size=3, color='#CC0000')))
            lb = LEB.get(2026)
            if lb:
                fv.add_shape(type='line', x0=lb, x1=lb, y0=0, y1=1, yref='paper',
                             line=dict(color='#00AA00', width=0.8, dash='dash'))
            lv = chart_base(185)
            lv['title'] = dict(text=f"{dl(kec)}  ·  MAPE {mv:.1f}%",
                               font=dict(size=9, color='#333'), x=0.5, xanchor='center')
            lv['margin'] = dict(l=6, r=6, t=24, b=60)
            lv['xaxis']['tickformat'] = '%b %Y'
            lv['xaxis']['tickfont']   = dict(size=7.5, color='#aaa')
            lv['yaxis']['tickfont']   = dict(size=7.5, color='#aaa')
            lv['legend'] = dict(orientation='h', y=-0.4, x=0.5, xanchor='center',
                                font=dict(size=7.5), bgcolor='rgba(255,255,255,0.9)')
            fv.update_layout(**lv)
            with col:
                st.plotly_chart(fv, use_container_width=True, config={'displayModeBar': False})

    with rc:
        st.markdown('<p class="sec">Metrik Validasi</p>', unsafe_allow_html=True)
        disp = dv[['District','MAPE (%)','RMSE','MAE','Bias']].sort_values('MAPE (%)')
        st.dataframe(
            disp.style
                .highlight_min(subset=['MAPE (%)'], color='#d4edda')
                .highlight_max(subset=['MAPE (%)'], color='#f8d7da')
                .format({'MAPE (%)':'{:.1f}%','RMSE':'{:,}','MAE':'{:,}','Bias':'{:+,}'}),
            use_container_width=True, hide_index=True, height=220)

        st.markdown('<p class="sec" style="margin-top:12px">Interpretasi</p>', unsafe_allow_html=True)
        for clr, bg, title, desc in [
            ("#16a34a","#f0fdf4","✅ Performa Baik","District A, C, E — MAPE di bawah 12%."),
            ("#d97706","#fffbeb","⚠️ Perlu Perhatian","District B & D melebihi target MAPE 15%."),
            ("#CC0000","#fff5f5","🔴 Efek Lebaran","Penurunan permintaan di luar rentang pelatihan."),
        ]:
            st.markdown(f"""
            <div class="icard" style="background:{bg};border-color:{clr}">
              <p class="ititle" style="color:{clr}">{title}</p>
              <p class="ibody">{desc}</p>
            </div>""", unsafe_allow_html=True)
