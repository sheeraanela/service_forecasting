import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="AHASS Prakiraan", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
* { font-family: 'DM Sans', sans-serif !important; }

#MainMenu, footer, header,
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, [data-testid="collapsedControl"],
[data-testid="stSidebar"] { display: none !important; }

.stApp { background: #f5f5f5 !important; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

.app-hdr {
    background: #CC0000; padding: 14px 0 12px 0;
    margin: 0 -2rem 1.2rem -2rem;
}
.app-hdr-title {
    color: #fff !important; font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase; margin: 0;
}
.app-hdr-sub { color: rgba(255,255,255,0.72) !important; font-size: 0.68rem; margin: 2px 0 0 0; }

[data-testid="stMetricLabel"] {
    font-size: 0.6rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.9px; color: #888 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important; font-weight: 700 !important; color: #111 !important;
}
[data-testid="metric-container"] {
    background: #fff; border-radius: 8px; padding: 12px 16px !important;
    border: 1px solid #e8e8e8;
}
[data-testid="stMetricDelta"] { display: none !important; }

div[data-testid="stSelectbox"] label {
    font-size: 0.62rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.9px !important; color: #888 !important;
}
div[data-testid="stSelectbox"] > div > div {
    font-size: 0.82rem !important; border-color: #ddd !important;
    border-radius: 6px !important; background: #fff !important;
}

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

.sec {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #888; border-bottom: 1.5px solid #CC0000;
    padding-bottom: 4px; margin: 0 0 10px 0;
}

.card {
    background: #fff; border-radius: 8px; padding: 14px 16px;
    border: 1px solid #e8e8e8;
}

.drow { display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #f2f2f2; }
.ddot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dname { font-size:0.75rem; font-weight:600; color:#111; }
.dmape { font-size:0.6rem; color:#bbb; margin-left:3px; }
.dval  { font-size:0.78rem; font-weight:700; }
.dsum  { font-size:0.58rem; color:#bbb; }

.frow { display:flex; align-items:center; gap:6px; margin-bottom:8px; }
.fpill { width:18px; height:18px; border-radius:4px; flex-shrink:0;
         display:flex; align-items:center; justify-content:center; font-size:0.6rem; font-weight:700; }
.fname { font-size:0.75rem; font-weight:700; color:#111; }
.fval  { font-size:0.6rem; color:#bbb; margin-left:3px; }
.fdesc { font-size:0.6rem; color:#777; }
.fbar  { height:3px; background:#eee; border-radius:2px; margin-top:3px; }
.fimp  { font-size:0.68rem; font-weight:700; width:40px; text-align:right; flex-shrink:0; }

.note { background:#fff5f5; border-left:2px solid #CC0000; padding:6px 10px;
        font-size:0.68rem; color:#555; margin-top:8px; border-radius:0 4px 4px 0; }

.icard { border-left:3px solid; padding:8px 12px; margin-bottom:8px; border-radius:0 6px 6px 0; }
.ititle { font-size:0.72rem; font-weight:700; margin:0 0 2px 0; }
.ibody  { font-size:0.65rem; color:#555; margin:0; }

.range-btn {
    display: inline-block; padding: 4px 12px; margin-right: 4px;
    border: 1.5px solid #ddd; border-radius: 4px; font-size: 0.7rem;
    font-weight: 600; cursor: pointer; background: #fff; color: #444;
}
.range-btn.active { background: #CC0000; color: #fff; border-color: #CC0000; }

[data-testid="column"] { padding: 0 6px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0.6rem !important; }

div[data-testid="stDateInput"] label {
    font-size: 0.62rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.9px !important; color: #888 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
FEAT = {
    'lag_1':'Jumlah servis 1 minggu lalu', 'lag_2':'Jumlah servis 2 minggu lalu',
    'lag_4':'Jumlah servis 4 minggu lalu (±1 bln)', 'lag_8':'Jumlah servis 8 minggu lalu (±2 bln)',
    'lag_13':'Jumlah servis 13 minggu lalu (±3 bln)', 'roll4_mean':'Rata-rata servis 4 minggu terakhir',
    'roll8_mean':'Rata-rata servis 8 minggu terakhir', 'roll13_mean':'Rata-rata servis 13 minggu terakhir',
    'roll4_std':'Volatilitas servis 4 minggu terakhir', 'roll8_std':'Volatilitas servis 8 minggu terakhir',
    'month':'Bulan (1–12)', 'quarter':'Kuartal (1–4)', 'year':'Tahun',
    'month_sin':'Pola musiman (sinus)', 'month_cos':'Pola musiman (kosinus)',
    'is_lebaran':'Indikator minggu Lebaran', 'is_holiday':'Indikator libur nasional',
    'is_christmas':'Indikator minggu Natal', 'is_newyear':'Indikator minggu Tahun Baru',
}
FEAT_DF = pd.DataFrame([{'Nama Fitur': k, 'Penjelasan': v} for k, v in FEAT.items()])

def fd(n): return FEAT.get(n, n)
def fl(n): return f"{n} — {fd(n)}"
def dl(r): return r.replace("DISTRICT_", "District ")

@st.cache_resource
def load():
    b = "streamlit_assets"
    with open(f"{b}/xgb_models.pkl", "rb") as f:  xgb_m = pickle.load(f)
    with open(f"{b}/forecast_results.pkl", "rb") as f: fcr = pickle.load(f)
    with open(f"{b}/shap_results.pkl", "rb") as f: shp = pickle.load(f)
    with open(f"{b}/config.pkl", "rb") as f: cfg = pickle.load(f)
    df = pd.read_csv(f"{b}/dfs_smooth.csv")
    df['Tanggal Servis'] = pd.to_datetime(df['Tanggal Servis'])
    for k in fcr:
        fcr[k]['Tanggal Servis'] = pd.to_datetime(fcr[k]['Tanggal Servis'])
    return xgb_m, fcr, shp, cfg, df

xgb_m, fcr, shp, cfg, hdf = load()
TOP5 = cfg['kecamatan']
COLS = cfg['warna_kec']
CM   = dict(zip(TOP5, COLS))
LEB  = {yr: v[0] for yr, v in cfg['lebaran'].items()}

def chart_base(h=320):
    return dict(
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        height=h, font=dict(family='DM Sans', size=9, color='#555'),
        margin=dict(l=8, r=8, t=8, b=80),
        xaxis=dict(showgrid=True, gridcolor='#f2f2f2', linecolor='#e0e0e0',
                   zeroline=False, tickfont=dict(size=8, color='#aaa')),
        yaxis=dict(showgrid=True, gridcolor='#f2f2f2', linecolor='#e0e0e0',
                   zeroline=False, tickfont=dict(size=8, color='#aaa')),
        legend=dict(orientation='h', y=-0.28, x=0.5, xanchor='center',
                    font=dict(size=8, color='#666'),
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#eee', borderwidth=1))

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-hdr">
  <div style="padding:0 2rem">
    <p class="app-hdr-title">AHASS Prakiraan Permintaan</p>
    <p class="app-hdr-sub">Prakiraan Servis Mingguan — Top 5 District Jakarta · XGBoost + SHAP XAI</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── CONTROLS ───────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 3, 4])
with c1:
    sel = st.selectbox("District", TOP5, format_func=dl)
with c2:
    fc = fcr[sel]
    mp = xgb_m[sel]['mape'] * 100
    st.markdown(
        f"<div style='padding:8px 0'>"
        f"<span style='font-size:0.8rem;color:#444'>"
        f"<b>Model:</b> XGBoost &nbsp;·&nbsp; <b>Pelatihan:</b> 2022–2024 &nbsp;·&nbsp; "
        f"<b>Uji:</b> 2025 &nbsp;·&nbsp; "
        f"<b>MAPE {dl(sel)}:</b> <span style='color:#CC0000;font-weight:700'>{mp:.1f}%</span>"
        f"</span></div>", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["  PRAKIRAAN  ", "  PENJELASAN XAI  ", "  PERBANDINGAN AKTUAL  "])

# ══════════════════════════════════════════════════════════════════════════════
with t1:
    # ── KPI per bulan ─────────────────────────────────────────────────────────
    st.markdown('<p class="sec">Ringkasan Prakiraan per Bulan</p>', unsafe_allow_html=True)

    fc_monthly = fc.copy()
    fc_monthly['bulan'] = fc_monthly['Tanggal Servis'].dt.to_period('M')
    monthly = fc_monthly.groupby('bulan').agg(
        total_forecast=('forecast', 'sum'),
        total_optimistic=('optimistic', 'sum'),
        total_pessimistic=('pessimistic', 'sum')
    ).reset_index()
    monthly['bulan_str'] = monthly['bulan'].astype(str)

    # Tampilkan 4 bulan pertama sebagai KPI
    cols_kpi = st.columns(min(len(monthly), 6))
    for i, (_, row) in enumerate(monthly.iterrows()):
        if i >= len(cols_kpi): break
        with cols_kpi[i]:
            try:
                label = pd.Period(row['bulan_str']).strftime('%b %Y')
            except:
                label = row['bulan_str']
            st.metric(label, f"{row['total_forecast']:,.0f}", help=f"Optimistis: {row['total_optimistic']:,.0f} | Pesimistis: {row['total_pessimistic']:,.0f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Filter rentang tanggal ────────────────────────────────────────────────
    st.markdown('<p class="sec">Filter Rentang Tampilan</p>', unsafe_allow_html=True)

    fc_min = fc['Tanggal Servis'].min().date()
    fc_max = fc['Tanggal Servis'].max().date()

    col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
    with col_f1:
        tgl_mulai = st.date_input("Dari Tanggal", value=fc_min, min_value=fc_min, max_value=fc_max)
    with col_f2:
        tgl_akhir = st.date_input("Sampai Tanggal", value=fc_max, min_value=fc_min, max_value=fc_max)
    with col_f3:
        st.markdown("<div style='padding-top:22px'>", unsafe_allow_html=True)
        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            if st.button("1 Minggu", use_container_width=True):
                tgl_mulai = fc_min
                tgl_akhir = (pd.Timestamp(fc_min) + pd.Timedelta(weeks=1)).date()
                st.rerun()
        with qc2:
            if st.button("2 Minggu", use_container_width=True):
                tgl_mulai = fc_min
                tgl_akhir = (pd.Timestamp(fc_min) + pd.Timedelta(weeks=2)).date()
                st.rerun()
        with qc3:
            if st.button("1 Bulan", use_container_width=True):
                tgl_mulai = fc_min
                tgl_akhir = (pd.Timestamp(fc_min) + pd.Timedelta(weeks=4)).date()
                st.rerun()
        with qc4:
            if st.button("Semua", use_container_width=True):
                tgl_mulai = fc_min
                tgl_akhir = fc_max
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Filter data berdasarkan tanggal
    mask_fc = (fc['Tanggal Servis'].dt.date >= tgl_mulai) & (fc['Tanggal Servis'].dt.date <= tgl_akhir)
    fc_filtered = fc[mask_fc]

    # Hitung berapa minggu yang ditampilkan
    n_minggu = len(fc_filtered)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Grafik utama ──────────────────────────────────────────────────────────
    gc, sc = st.columns([3, 1])

    with gc:
        st.markdown(f'<p class="sec">Grafik Prakiraan Mingguan ({n_minggu} Minggu · {tgl_mulai.strftime("%d %b %Y")} – {tgl_akhir.strftime("%d %b %Y")})</p>', unsafe_allow_html=True)

        # Data historis — ambil semua untuk konteks, tapi highlight sesuai filter
        h = hdf[hdf['Kecamatan Bengkel'] == sel].copy()

        # Lebar grafik dinamis berdasarkan jumlah minggu
        # Minimal 400px, tiap minggu 40px, maksimal scroll
        chart_width = max(600, n_minggu * 45)

        fds = fc_filtered['Tanggal Servis'].astype(str)
        fig = go.Figure()

        # Historis 2025 (semua, sebagai konteks)
        h_all = hdf[hdf['Kecamatan Bengkel'] == sel].copy()
        fig.add_trace(go.Bar(
            x=h_all['Tanggal Servis'].astype(str), y=h_all['Jumlah Servis'],
            name='Historis', marker_color='#DAEAF8', marker_line_width=0, opacity=0.7))

        # Rentang kepercayaan
        fig.add_trace(go.Scatter(
            x=pd.concat([fds, fds[::-1]]),
            y=pd.concat([fc_filtered['optimistic'], fc_filtered['pessimistic'][::-1]]),
            fill='toself', fillcolor='rgba(204,0,0,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))

        # Garis pesimistis dan optimistis
        fig.add_trace(go.Scatter(
            x=fds, y=fc_filtered['pessimistic'], mode='lines', name='Pesimistis −16%',
            line=dict(color='#CCC', dash='dash', width=1.2)))
        fig.add_trace(go.Scatter(
            x=fds, y=fc_filtered['optimistic'], mode='lines', name='Optimistis +18%',
            line=dict(color='#3B6D11', dash='dash', width=1.2)))

        # Garis prakiraan utama
        fig.add_trace(go.Scatter(
            x=fds, y=fc_filtered['forecast'], mode='lines+markers', name='Prakiraan (XGBoost)',
            line=dict(color=CM[sel], width=2.5),
            marker=dict(size=5, color=CM[sel])))

        # Label nilai di tiap titik prakiraan (kalau tidak terlalu banyak)
        if n_minggu <= 12:
            fig.add_trace(go.Scatter(
                x=fds, y=fc_filtered['forecast'],
                mode='text',
                text=[f"{v:,.0f}" for v in fc_filtered['forecast']],
                textposition='top center',
                textfont=dict(size=8, color=CM[sel]),
                showlegend=False))

        # Garis Lebaran 2026
        lb = LEB.get(2026)
        if lb:
            lb_str = str(pd.Timestamp(lb).date())
            if tgl_mulai.strftime('%Y-%m-%d') <= lb_str <= tgl_akhir.strftime('%Y-%m-%d'):
                fig.add_shape(type='line', x0=lb_str, x1=lb_str, y0=0, y1=1, yref='paper',
                              line=dict(color='#00AA00', width=1.5, dash='dash'))
                fig.add_annotation(x=lb_str, y=1.04, yref='paper', text='Lebaran 2026',
                                   showarrow=False, font=dict(size=8, color='#00AA00'), xanchor='center')

        layout = chart_base(320)
        layout['xaxis']['tickformat'] = '%d %b %Y'
        layout['xaxis']['tickangle'] = -35
        layout['yaxis']['title'] = 'Servis/Minggu'
        layout['yaxis']['title_font'] = dict(size=9, color='#bbb')

        # Aktifkan scroll horizontal
        layout['xaxis']['rangeslider'] = dict(visible=True, thickness=0.05, bgcolor='#f5f5f5')
        layout['height'] = 380

        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d'],
            'scrollZoom': True
        })

        # Tabel detail mingguan
        st.markdown('<p class="sec" style="margin-top:8px">Detail per Minggu</p>', unsafe_allow_html=True)
        tbl = fc_filtered[['Tanggal Servis', 'forecast', 'optimistic', 'pessimistic']].copy()
        tbl['Tanggal Servis'] = tbl['Tanggal Servis'].dt.strftime('%d %b %Y')
        tbl.columns = ['Minggu', 'Prakiraan', 'Optimistis (+18%)', 'Pesimistis (−16%)']
        tbl = tbl.reset_index(drop=True)
        st.dataframe(
            tbl.style.format({
                'Prakiraan': '{:,.0f}',
                'Optimistis (+18%)': '{:,.0f}',
                'Pesimistis (−16%)': '{:,.0f}'
            }).background_gradient(subset=['Prakiraan'], cmap='Reds', low=0.4),
            use_container_width=True, hide_index=True, height=200)

    with sc:
        st.markdown('<p class="sec">Semua District</p>', unsafe_allow_html=True)
        for kec, clr in zip(TOP5, COLS):
            fk = fcr[kec]
            mask_k = (fk['Tanggal Servis'].dt.date >= tgl_mulai) & (fk['Tanggal Servis'].dt.date <= tgl_akhir)
            fk_f = fk[mask_k]
            avg_fc = fk_f['forecast'].mean() if len(fk_f) > 0 else 0
            sum_fc = fk_f['forecast'].sum() if len(fk_f) > 0 else 0
            mape_v = xgb_m[kec]['mape'] * 100
            st.markdown(f"""
            <div class="drow">
              <div class="ddot" style="background:{clr}"></div>
              <div style="flex:1">
                <span class="dname">{dl(kec)}</span>
                <span class="dmape">MAPE {mape_v:.1f}%</span>
              </div>
              <div style="text-align:right">
                <div class="dval" style="color:{clr}">{avg_fc:,.0f}</div>
                <div class="dsum">∑ {sum_fc:,.0f}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="sec">Ringkasan Rentang</p>', unsafe_allow_html=True)
        total_all = sum(
            fcr[kec][
                (fcr[kec]['Tanggal Servis'].dt.date >= tgl_mulai) &
                (fcr[kec]['Tanggal Servis'].dt.date <= tgl_akhir)
            ]['forecast'].sum()
            for kec in TOP5
        )
        st.markdown(f"""
        <div style="background:#fff5f5;border-left:3px solid #CC0000;padding:10px 12px;border-radius:0 6px 6px 0;margin-top:4px">
          <div style="font-size:0.6rem;color:#888;font-weight:700;text-transform:uppercase;letter-spacing:1px">Total 5 District</div>
          <div style="font-size:1.4rem;font-weight:700;color:#CC0000">{total_all:,.0f}</div>
          <div style="font-size:0.6rem;color:#bbb">{n_minggu} minggu · {tgl_mulai.strftime('%d %b')} – {tgl_akhir.strftime('%d %b %Y')}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with t2:
    # Pilih minggu untuk XAI
    wo  = [f"Minggu {i+1} — {d.strftime('%d %b %Y')}" for i, d in enumerate(fc['Tanggal Servis'])]
    col_xai1, col_xai2 = st.columns([2, 5])
    with col_xai1:
        sw  = st.selectbox("Pilih Minggu", wo, key="xai_week")
        wi  = wo.index(sw)

    xai = fc.iloc[wi]
    sr  = shp[sel]
    ii  = min(wi, len(sr['shap_values']) - 1)
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
            f"{xai['Tanggal Servis'].strftime('%d %b %Y')} · {dl(sel)}</p>", unsafe_allow_html=True)

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
    st.markdown('<p class="sec">Upload Data Aktual</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.72rem;color:#888;margin:0 0 6px 0'>"
        "Upload file CSV dengan kolom: <code>Kecamatan</code>, <code>Tanggal Servis</code>, <code>Permintaan Servis</code></p>",
        unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="hidden")

    def proses_aktual(file_obj):
        try:
            df = pd.read_csv(file_obj)
            # Validasi kolom
            required = ['Kecamatan', 'Tanggal Servis', 'Permintaan Servis']
            missing  = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Kolom tidak ditemukan: {missing}")
                return None, None
            df['Tanggal Servis']    = pd.to_datetime(df['Tanggal Servis'])
            df['Permintaan Servis'] = pd.to_numeric(df['Permintaan Servis'], errors='coerce')
            df = df.dropna(subset=['Permintaan Servis'])

            # Hitung metrik per kecamatan
            rows = []
            for kec in TOP5:
                act = df[df['Kecamatan'] == kec][['Tanggal Servis', 'Permintaan Servis']].copy()
                if act.empty:
                    continue
                act = act.rename(columns={'Tanggal Servis': 'ds', 'Permintaan Servis': 'y_actual'})
                merged = pd.merge(
                    fcr[kec][['Tanggal Servis', 'forecast']].rename(columns={'Tanggal Servis': 'ds'}),
                    act, on='ds', how='inner')
                merged = merged[merged['y_actual'] > 0]
                if merged.empty:
                    continue
                fc_v = merged['forecast'].values
                ac_v = merged['y_actual'].values
                rows.append({
                    'Kecamatan': kec,
                    'District':  dl(kec),
                    'MAPE (%)':  round(np.mean(np.abs(fc_v - ac_v) / ac_v) * 100, 1),
                    'MAE':       int(np.mean(np.abs(fc_v - ac_v))),
                    'RMSE':      int(np.sqrt(np.mean((fc_v - ac_v) ** 2))),
                    'Bias':      int(np.mean(fc_v - ac_v)),
                    'Minggu':    len(merged),
                    '_merged':   merged,
                })
            return df, rows
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return None, None

    if uploaded is not None:
        df_actual, hasil = proses_aktual(uploaded)

        if hasil:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # KPI strip
            mapes = [r['MAPE (%)'] for r in hasil]
            best  = hasil[np.argmin(mapes)]
            worst = hasil[np.argmax(mapes)]
            tgl_min = df_actual['Tanggal Servis'].min().strftime('%d %b %Y')
            tgl_max = df_actual['Tanggal Servis'].max().strftime('%d %b %Y')

            k1, k2, k3, k4 = st.columns(4)
            with k1: st.metric("Rata-rata MAPE", f"{np.mean(mapes):.1f}%")
            with k2: st.metric("Terbaik",  dl(best['Kecamatan']),  help=f"MAPE {best['MAPE (%)']:.1f}%")
            with k3: st.metric("Terlemah", dl(worst['Kecamatan']), help=f"MAPE {worst['MAPE (%)']:.1f}%")
            with k4: st.metric("Periode",  f"{tgl_min} – {tgl_max}")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            lc, rc = st.columns([2.4, 1])

            with lc:
                # Pilih district untuk grafik perbandingan
                sel_v = st.selectbox("District untuk perbandingan", TOP5, format_func=dl, key="val_sel")
                row_v = next((r for r in hasil if r['Kecamatan'] == sel_v), None)

                if row_v:
                    st.markdown(f'<p class="sec">Prakiraan vs Aktual — {dl(sel_v)}</p>', unsafe_allow_html=True)

                    merged = row_v['_merged']
                    clr    = CM[sel_v]
                    fck    = fcr[sel_v]
                    hk     = hdf[hdf['Kecamatan Bengkel'] == sel_v].tail(26)

                    fv = go.Figure()
                    fv.add_trace(go.Bar(
                        x=hk['Tanggal Servis'].astype(str), y=hk['Jumlah Servis'],
                        name='Historis 2025', marker_color='#DAEAF8',
                        marker_line_width=0, opacity=0.75))
                    fv.add_trace(go.Scatter(
                        x=pd.concat([fck['Tanggal Servis'].astype(str),
                                     fck['Tanggal Servis'].astype(str)[::-1]]),
                        y=pd.concat([fck['optimistic'], fck['pessimistic'][::-1]]),
                        fill='toself', fillcolor='rgba(204,0,0,0.06)',
                        line=dict(color='rgba(0,0,0,0)'), name='Rentang Kepercayaan'))
                    fv.add_trace(go.Scatter(
                        x=fck['Tanggal Servis'].astype(str), y=fck['forecast'],
                        mode='lines', name='Prakiraan (XGBoost)',
                        line=dict(color=clr, width=2)))
                    fv.add_trace(go.Scatter(
                        x=merged['ds'].astype(str), y=merged['y_actual'],
                        mode='lines+markers', name='Aktual',
                        line=dict(color='#CC0000', width=2.2),
                        marker=dict(size=5, color='#CC0000')))

                    lb = LEB.get(2026)
                    if lb:
                        fv.add_shape(type='line', x0=str(lb), x1=str(lb), y0=0, y1=1, yref='paper',
                                     line=dict(color='#00AA00', width=1, dash='dash'))
                        fv.add_annotation(x=str(lb), y=1.04, yref='paper', text='Lebaran 2026',
                                          showarrow=False, font=dict(size=8, color='#00AA00'), xanchor='center')

                    lv = chart_base(300)
                    lv['xaxis']['tickformat']  = '%b %Y'
                    lv['xaxis']['rangeslider'] = dict(visible=True, thickness=0.05, bgcolor='#f5f5f5')
                    lv['yaxis']['title']       = 'Servis / Minggu'
                    lv['yaxis']['title_font']  = dict(size=9, color='#bbb')
                    lv['height'] = 360
                    fv.update_layout(**lv)
                    st.plotly_chart(fv, use_container_width=True, config={
                        'displayModeBar': True, 'scrollZoom': True,
                        'modeBarButtonsToRemove': ['select2d', 'lasso2d']})

                    # Metrik detail
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("MAPE",  f"{row_v['MAPE (%)']:.1f}%")
                    with m2: st.metric("RMSE",  f"{row_v['RMSE']:,}")
                    with m3: st.metric("MAE",   f"{row_v['MAE']:,}")
                    with m4: st.metric("Bias",  f"{row_v['Bias']:+,}", help="+ = model terlalu tinggi · − = model terlalu rendah")

                    # Tabel detail per minggu
                    st.markdown('<p class="sec" style="margin-top:8px">Detail per Minggu</p>', unsafe_allow_html=True)
                    tbl_v = merged[['ds', 'forecast', 'y_actual']].copy()
                    tbl_v['selisih']  = tbl_v['forecast'] - tbl_v['y_actual']
                    tbl_v['mape_w']   = np.abs(tbl_v['selisih'] / tbl_v['y_actual']) * 100
                    tbl_v['ds']       = tbl_v['ds'].dt.strftime('%d %b %Y')
                    tbl_v.columns     = ['Minggu', 'Prakiraan', 'Aktual', 'Selisih', 'MAPE (%)']
                    st.dataframe(
                        tbl_v.style.format({
                            'Prakiraan': '{:,.0f}', 'Aktual': '{:,.0f}',
                            'Selisih': '{:+,.0f}', 'MAPE (%)': '{:.1f}%'
                        }).background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r'),
                        use_container_width=True, hide_index=True, height=200)

            with rc:
                st.markdown('<p class="sec">Metrik per District</p>', unsafe_allow_html=True)
                tbl_m = pd.DataFrame([{
                    'District':  dl(r['Kecamatan']),
                    'MAPE (%)':  r['MAPE (%)'],
                    'RMSE':      r['RMSE'],
                    'MAE':       r['MAE'],
                    'Bias':      r['Bias'],
                    'Minggu':    r['Minggu'],
                } for r in hasil]).sort_values('MAPE (%)')
                st.dataframe(
                    tbl_m.style
                        .highlight_min(subset=['MAPE (%)'], color='#d4edda')
                        .highlight_max(subset=['MAPE (%)'], color='#f8d7da')
                        .format({'MAPE (%)': '{:.1f}%', 'RMSE': '{:,}',
                                 'MAE': '{:,}', 'Bias': '{:+,}'}),
                    use_container_width=True, hide_index=True, height=240)

                st.markdown('<p class="sec" style="margin-top:12px">Interpretasi</p>', unsafe_allow_html=True)
                for clr2, bg, title, desc in [
                    ("#16a34a", "#f0fdf4", "✅ Performa Baik",     "MAPE di bawah 12% — prediksi akurat."),
                    ("#d97706", "#fffbeb", "⚠️ Perlu Perhatian", "MAPE 12–20% — perlu evaluasi lebih lanjut."),
                    ("#CC0000", "#fff5f5", "🔴 Distribusi Shift",  "Pola demand berbeda dari data training."),
                ]:
                    st.markdown(f"""
                    <div class="icard" style="background:{bg};border-color:{clr2}">
                      <p class="ititle" style="color:{clr2}">{title}</p>
                      <p class="ibody">{desc}</p>
                    </div>""", unsafe_allow_html=True)
    else:
        # Tampilan kosong sebelum upload
        st.markdown("""
        <div style="background:#fff;border:2px dashed #e8e8e8;border-radius:8px;
                    padding:40px;text-align:center;margin-top:16px">
          <div style="font-size:2rem;margin-bottom:8px">📂</div>
          <div style="font-size:0.9rem;font-weight:600;color:#444;margin-bottom:4px">
            Upload file CSV untuk mulai perbandingan
          </div>
          <div style="font-size:0.72rem;color:#bbb">
            Kolom yang dibutuhkan: <code>Kecamatan</code>, <code>Tanggal Servis</code>, <code>Permintaan Servis</code>
          </div>
        </div>""", unsafe_allow_html=True)

        with st.expander("💡 Cara export data aktual dari Google Colab"):
            st.code("""# Export data aktual dari notebook
rows = []
for kec in kecamatan:
    sub = dfs[dfs['Kecamatan Bengkel'] == kec][['Tanggal Servis', 'Jumlah Servis']].copy()
    sub['Kecamatan'] = kec
    sub = sub.rename(columns={'Jumlah Servis': 'Permintaan Servis'})
    rows.append(sub)

df_export = pd.concat(rows)
df_export['Tanggal Servis'] = df_export['Tanggal Servis'].dt.strftime('%Y-%m-%d')
df_export.to_csv('/content/data_aktual.csv', index=False)

from google.colab import files
files.download('/content/data_aktual.csv')""", language="python")
