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

*, html, body { font-family: 'Inter', sans-serif !important; box-sizing: border-box; margin: 0; padding: 0; }
html, body { overflow: hidden !important; height: 100vh; }
.stApp { background: #fff !important; overflow: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; overflow: hidden !important; }
section[data-testid="stMain"] { overflow: hidden !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stAppViewBlockContainer"] { padding: 0 !important; }

/* Hide chrome */
#MainMenu, footer, header, [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stDeployButton,
[data-testid="collapsedControl"], [data-testid="stSidebar"] { display: none !important; }

/* ── Header ── */
.hdr { background: #CC0000; padding: 10px 24px; display: flex; align-items: center; justify-content: space-between; }
.hdr-t { color: #fff !important; font-size: 1.05rem; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
.hdr-s { color: rgba(255,255,255,0.7) !important; font-size: 0.65rem; margin-top: 1px; }

/* ── Controls ── */
.ctrl { padding: 6px 24px; background: #fff; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 12px; }
div[data-testid="stSelectbox"] label {
    font-size: 0.58rem !important; font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.9px !important; color: #999 !important; line-height: 1 !important; margin-bottom: 2px !important;
}
div[data-testid="stSelectbox"] > div > div {
    min-height: 28px !important; font-size: 0.78rem !important; border-color: #ddd !important;
    border-radius: 3px !important; padding: 2px 8px !important; color: #111 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; padding: 0 24px; border-bottom: 2px solid #CC0000 !important; background: #fff; flex-shrink: 0;
}
.stTabs [data-baseweb="tab"] {
    border: 1.5px solid #222; border-bottom: none; background: #fff;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.4px;
    padding: 5px 16px; border-radius: 0; color: #222 !important;
}
.stTabs [aria-selected="true"] { background: #CC0000 !important; color: #fff !important; border-color: #CC0000 !important; }
[data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 12px 24px 0 24px !important; overflow-y: auto !important; }

/* ── Metrics ── */
[data-testid="stMetricLabel"] {
    font-size: 0.57rem !important; font-weight: 700 !important; text-transform: uppercase;
    letter-spacing: 0.9px; color: #999 !important; margin: 0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important; font-weight: 700 !important; color: #111 !important; line-height: 1.15 !important;
}
[data-testid="metric-container"] { background: none !important; padding: 0 !important; }
[data-testid="stMetricDelta"] { font-size: 0.62rem !important; }

/* ── Section heading ── */
.sh { font-size: 0.6rem; font-weight: 700; letter-spacing: 1.1px; text-transform: uppercase;
      border-bottom: 1.5px solid #CC0000; padding-bottom: 3px; margin: 10px 0 6px 0; color: #111; }

/* ── District row ── */
.dr { display:flex; align-items:center; gap:8px; padding:5px 0; border-bottom:1px solid #f5f5f5; }
.dr-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.dr-name { font-size:0.72rem; font-weight:600; color:#111; }
.dr-mape { font-size:0.58rem; color:#bbb; margin-left:3px; }
.dr-val  { font-size:0.75rem; font-weight:700; }
.dr-sum  { font-size:0.55rem; color:#bbb; }

/* ── Feature row (XAI) ── */
.fr { display:flex; align-items:center; gap:6px; margin-bottom:7px; }
.fp { width:16px; height:16px; border-radius:3px; display:flex; align-items:center;
      justify-content:center; font-size:0.55rem; font-weight:700; flex-shrink:0; }
.fn { font-size:0.72rem; font-weight:700; color:#111; }
.fv { font-size:0.58rem; color:#bbb; margin-left:3px; }
.fd { font-size:0.58rem; color:#777; margin-top:1px; }
.fb { height:3px; background:#eee; border-radius:2px; margin-top:3px; }
.fi { font-size:0.65rem; font-weight:700; width:38px; text-align:right; flex-shrink:0; }

/* ── Note ── */
.note { background:#fafafa; border-left:2px solid #CC0000; padding:5px 8px; font-size:0.65rem; color:#555; margin-top:6px; }

/* ── Interp card ── */
.ic { border-left:2px solid; padding:5px 10px; margin-bottom:6px; border-radius:0 5px 5px 0; }
.ic-t { font-size:0.7rem; font-weight:700; margin:0 0 1px 0; }
.ic-b { font-size:0.62rem; color:#555; margin:0; }

/* ── Columns ── */
[data-testid="column"] { padding: 0 5px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

/* ── Divider ── */
.div { border:none; border-top:1px solid #f0f0f0; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────────────────────────
FEAT = {
    'lag_1':'Jumlah servis 1 minggu lalu','lag_2':'Jumlah servis 2 minggu lalu',
    'lag_4':'Jumlah servis 4 minggu lalu (±1 bln)','lag_8':'Jumlah servis 8 minggu lalu (±2 bln)',
    'lag_13':'Jumlah servis 13 minggu lalu (±3 bln)','roll4_mean':'Rata-rata servis 4 minggu terakhir',
    'roll8_mean':'Rata-rata servis 8 minggu terakhir','roll13_mean':'Rata-rata servis 13 minggu terakhir',
    'roll4_std':'Volatilitas servis 4 minggu terakhir','roll8_std':'Volatilitas servis 8 minggu terakhir',
    'month':'Bulan (1–12)','quarter':'Kuartal (1–4)','year':'Tahun',
    'month_sin':'Pola musiman (sinus)','month_cos':'Pola musiman (kosinus)',
    'is_lebaran':'Indikator minggu Lebaran','is_holiday':'Indikator libur nasional',
}
FEAT_DF = pd.DataFrame([{'Nama Fitur':k,'Penjelasan':v} for k,v in FEAT.items()])
def fd(n): return FEAT.get(n,n)
def fl(n): return f"{n} — {fd(n)}"
def dl(r): return r.replace("KECAMATAN_","District ")

@st.cache_resource
def load():
    b="streamlit_assets"
    with open(f"{b}/lgb_models.pkl","rb") as f: lgb=pickle.load(f)
    with open(f"{b}/forecast_results.pkl","rb") as f: fcr=pickle.load(f)
    with open(f"{b}/shap_results.pkl","rb") as f: shp=pickle.load(f)
    with open(f"{b}/config.pkl","rb") as f: cfg=pickle.load(f)
    df=pd.read_csv(f"{b}/dfs_smooth.csv"); df['Tanggal Servis']=pd.to_datetime(df['Tanggal Servis'])
    dv=pd.read_csv(f"{b}/validation_metrics.csv")
    for k in fcr: fcr[k]['ds']=pd.to_datetime(fcr[k]['ds'])
    return lgb,fcr,shp,cfg,df,dv

lgb,fcr,shp,cfg,hdf,dv=load()
TOP5=cfg['TOP5']; COLS=cfg['KEC_COLORS']; CM=dict(zip(TOP5,COLS))
LEB={yr:v[0] for yr,v in cfg['LEBARAN'].items()}
dv=dv.copy(); dv['District']=dv['Kecamatan'].apply(dl)
vmape=dict(zip(dv['Kecamatan'],dv['MAPE (%)']))

def actuals(kec):
    fc=shp[kec]['feat_cols']; xt=shp[kec]['X_test']; li=fc.index('lag_1')
    return np.array([xt[i+1,li] if i+1<len(xt) else np.nan for i in range(13)])

def pb(h=220,**kw):
    d=dict(plot_bgcolor='#fff',paper_bgcolor='#fff',
           font=dict(family='Inter',color='#111',size=9),
           height=h, margin=dict(l=6,r=6,t=6,b=60),
           xaxis=dict(showgrid=True,gridcolor='#f5f5f5',linecolor='#e8e8e8',
                      tickfont=dict(size=8,color='#aaa'),zeroline=False),
           yaxis=dict(showgrid=True,gridcolor='#f5f5f5',linecolor='#e8e8e8',
                      tickfont=dict(size=8,color='#aaa'),zeroline=False),
           legend=dict(orientation='h',y=-0.3,x=0.5,xanchor='center',
                       font=dict(size=7.5,color='#555'),
                       bgcolor='rgba(255,255,255,0.9)',bordercolor='#eee',borderwidth=1))
    d.update(kw); return d

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div>
    <div class="hdr-t">AHASS Prakiraan Permintaan</div>
    <div class="hdr-s">Prakiraan Servis Mingguan — Top 5 District Jakarta · LightGBM + SHAP XAI</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── CONTROLS ──────────────────────────────────────────────────────────────────
st.markdown('<div class="ctrl">', unsafe_allow_html=True)
c1,c2,c3=st.columns([2,4,3])
with c1: sel=st.selectbox("District",TOP5,format_func=dl)
with c2:
    fc=fcr[sel]; wo=[f"Minggu {i+1} — {d.strftime('%d %b %Y')}" for i,d in enumerate(fc['ds'])]
    sw=st.selectbox("Minggu Prakiraan",wo); wi=wo.index(sw)
with c3:
    mp=lgb[sel]['mape']*100
    st.markdown(f"<span style='font-size:0.75rem;color:#444'>"
                f"<strong>Model:</strong> LightGBM &nbsp;·&nbsp; <strong>Pelatihan:</strong> 2022–2024 "
                f"&nbsp;·&nbsp; <strong>Uji:</strong> 2025 &nbsp;·&nbsp; "
                f"<strong>MAPE {dl(sel)}:</strong> <span style='color:#CC0000;font-weight:700'>{mp:.1f}%</span>"
                f"</span>",unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
row=fc.iloc[wi]
t1,t2,t3=st.tabs(["  PRAKIRAAN  ","  PENJELASAN XAI  ","  VALIDASI 2026  "])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with t1:
    # KPI
    k1,k2,k3,k4=st.columns(4)
    with k1: st.metric("District",dl(sel))
    with k2: st.metric("Prakiraan Dasar",f"{row['forecast']:,.0f}")
    with k3: st.metric("Optimistis +18%",f"{row['optimistic']:,.0f}")
    with k4: st.metric("Pesimistis -16%",f"{row['pessimistic']:,.0f}")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    gc,sc=st.columns([3,1])
    with gc:
        st.markdown('<div class="sh">Grafik Prakiraan 52 Minggu</div>', unsafe_allow_html=True)
        h=hdf[hdf['Kecamatan Bengkel']==sel].tail(26)
        fds=fc['ds'].astype(str)
        fig=go.Figure()
        fig.add_trace(go.Bar(x=h['Tanggal Servis'].astype(str),y=h['Jumlah Servis'],
                             name='Historis 2025',marker_color='#DAEAF8',marker_line_width=0,opacity=0.9))
        fig.add_trace(go.Scatter(x=pd.concat([fds,fds[::-1]]),
                                 y=pd.concat([fc['optimistic'],fc['pessimistic'][::-1]]),
                                 fill='toself',fillcolor='rgba(204,0,0,0.06)',
                                 line=dict(color='rgba(0,0,0,0)'),name='Rentang Kepercayaan'))
        fig.add_trace(go.Scatter(x=fds,y=fc['pessimistic'],mode='lines',name='Pesimistis',
                                 line=dict(color='#CCC',dash='dash',width=1)))
        fig.add_trace(go.Scatter(x=fds,y=fc['optimistic'],mode='lines',name='Optimistis',
                                 line=dict(color='#3B6D11',dash='dash',width=1)))
        fig.add_trace(go.Scatter(x=fds,y=fc['forecast'],mode='lines',name='Prakiraan (LightGBM)',
                                 line=dict(color=CM[sel],width=2)))
        lb=LEB.get(2026)
        if lb:
            fig.add_shape(type='line',x0=lb,x1=lb,y0=0,y1=1,yref='paper',
                          line=dict(color='#00AA00',width=1,dash='dash'))
            fig.add_annotation(x=lb,y=1.04,yref='paper',text='Lebaran',showarrow=False,
                               font=dict(size=7.5,color='#00AA00'),xanchor='center')
        fig.add_shape(type='line',x0=str(fc['ds'].iloc[wi]),x1=str(fc['ds'].iloc[wi]),
                      y0=0,y1=1,yref='paper',line=dict(color='#CC0000',width=1,dash='dot'))
        fig.update_layout(**pb(280,margin=dict(l=6,r=6,t=6,b=70),
            xaxis=dict(tickformat='%b %Y',showgrid=True,gridcolor='#f5f5f5',
                       linecolor='#e8e8e8',tickfont=dict(size=8,color='#aaa')),
            yaxis=dict(title='Servis/Minggu',showgrid=True,gridcolor='#f5f5f5',
                       linecolor='#e8e8e8',tickfont=dict(size=8,color='#aaa'),
                       title_font=dict(size=8,color='#ccc'))))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

    with sc:
        st.markdown('<div class="sh">Semua District</div>', unsafe_allow_html=True)
        for kec,clr in zip(TOP5,COLS):
            fk=fcr[kec]
            st.markdown(f"""
            <div class="dr">
              <div class="dr-dot" style="background:{clr}"></div>
              <div style="flex:1;min-width:0">
                <span class="dr-name">{dl(kec)}</span>
                <span class="dr-mape">MAPE {lgb[kec]['mape']*100:.1f}%</span>
              </div>
              <div style="text-align:right">
                <div class="dr-val" style="color:{clr}">{fk['forecast'].mean():,.0f}</div>
                <div class="dr-sum">∑ {fk['forecast'].sum():,.0f}/thn</div>
              </div>
            </div>""",unsafe_allow_html=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with t2:
    xai=fcr[sel].iloc[wi]; sr=shp[sel]
    ii=min(wi,len(sr['shap_values'])-1)
    sv,xv=sr['shap_values'][ii],sr['X_test'][ii]
    sdf=pd.DataFrame({'feature':sr['feat_cols'],'shap':sv,'value':xv})
    sdf['abs']=sdf['shap'].abs()
    top5f=sdf.sort_values('abs',ascending=False).head(5); mx=top5f['abs'].max()

    a,b,c=st.columns([1,1.4,1.6])

    with a:
        st.markdown('<div class="sh">5 Fitur Paling Berpengaruh</div>', unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.75rem;font-weight:700;margin:0 0 1px 0'>"
                    f"Prakiraan: <span style='color:#CC0000'>{xai['forecast']:,.0f}</span> servis/minggu</p>"
                    f"<p style='font-size:0.62rem;color:#bbb;margin:0 0 8px 0'>"
                    f"{xai['ds'].strftime('%d %b %Y')} · {dl(sel)}</p>",unsafe_allow_html=True)
        for _,r in top5f.iterrows():
            pos=r['shap']>0; clr="#CC0000" if pos else "#0055BB"
            bg="#FFF0F0" if pos else "#EEF4FF"; fill=int((r['abs']/mx)*100)
            st.markdown(f"""
            <div class="fr">
              <div class="fp" style="background:{bg};color:{clr}">{"▲" if pos else "▼"}</div>
              <div style="flex:1;min-width:0">
                <div><span class="fn">{r['feature']}</span><span class="fv">= {r['value']:.1f}</span></div>
                <div class="fd">{fd(r['feature'])}</div>
                <div class="fb"><div style="height:3px;border-radius:2px;width:{fill}%;background:{clr};opacity:0.55"></div></div>
              </div>
              <span class="fi" style="color:{clr}">{r['shap']:+.1f}</span>
            </div>""",unsafe_allow_html=True)
        st.markdown('<div class="note">▲ Merah = menaikkan · ▼ Biru = menurunkan · Lebar = dampak</div>',
                    unsafe_allow_html=True)

    with b:
        st.markdown(f'<div class="sh">Kepentingan Fitur — {dl(sel)}</div>', unsafe_allow_html=True)
        ms=np.abs(sr['shap_values']).mean(axis=0)
        df2=pd.DataFrame({'F':[fl(f) for f in sr['feat_cols']],'V':ms}).sort_values('V')
        f2=go.Figure(go.Bar(x=df2['V'],y=df2['F'],orientation='h',
                            marker_color='#CC0000',opacity=0.75,marker_line_width=0))
        f2.update_layout(**pb(340,margin=dict(l=6,r=6,t=6,b=20),
            xaxis=dict(title='|SHAP|',showgrid=True,gridcolor='#f5f5f5',linecolor='#e8e8e8',
                       tickfont=dict(size=7.5,color='#aaa'),title_font=dict(size=8,color='#bbb')),
            yaxis=dict(showgrid=False,tickfont=dict(size=7,color='#444')),
            legend=dict(orientation='h',y=-0.04)))
        st.plotly_chart(f2,use_container_width=True,config={'displayModeBar':False})

    with c:
        st.markdown('<div class="sh">SHAP Semua District</div>', unsafe_allow_html=True)
        f3=go.Figure()
        for kec,clr in zip(TOP5,COLS):
            sr2=shp[kec]; ms2=np.abs(sr2['shap_values']).mean(axis=0)
            f3.add_trace(go.Bar(name=dl(kec),x=[fl(f) for f in sr2['feat_cols']],
                                y=ms2,marker_color=clr,marker_line_width=0))
        f3.update_layout(**pb(200,barmode='group',margin=dict(l=6,r=6,t=6,b=130),
            xaxis=dict(tickangle=-35,showgrid=False,tickfont=dict(size=7,color='#888')),
            yaxis=dict(title='|SHAP|',showgrid=True,gridcolor='#f5f5f5',
                       tickfont=dict(size=7.5,color='#aaa'),title_font=dict(size=8,color='#bbb'))))
        st.plotly_chart(f3,use_container_width=True,config={'displayModeBar':False})

        st.markdown('<div class="sh">Kamus Fitur</div>', unsafe_allow_html=True)
        st.dataframe(FEAT_DF,use_container_width=True,hide_index=True,height=148)

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with t3:
    am=dv['MAPE (%)'].mean(); best=dv.loc[dv['MAPE (%)'].idxmin()]; worst=dv.loc[dv['MAPE (%)'].idxmax()]

    k1,k2,k3,k4=st.columns(4)
    with k1: st.metric("Rata-rata MAPE",f"{am:.1f}%")
    with k2: st.metric("Terbaik",dl(best['Kecamatan']),delta=f"MAPE {best['MAPE (%)']:.1f}%")
    with k3: st.metric("Terlemah",dl(worst['Kecamatan']),delta=f"MAPE {worst['MAPE (%)']:.1f}%",delta_color="inverse")
    with k4: st.metric("Periode","Jan – Mar 2026")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    lc,rc=st.columns([2.4,1])

    with lc:
        st.markdown('<div class="sh">Prakiraan vs Aktual 2026 per District</div>', unsafe_allow_html=True)
        r1=st.columns(3); r2=st.columns(3)
        for i,(kec,clr) in enumerate(zip(TOP5,COLS)):
            col=r1[i] if i<3 else r2[i-3]
            fck=fcr[kec]; hk=hdf[hdf['Kecamatan Bengkel']==kec].tail(26)
            fds=fck['ds'].astype(str); act=actuals(kec); ds13=fck['ds'].iloc[:13].astype(str)
            mv=vmape.get(kec,0)
            fv=go.Figure()
            fv.add_trace(go.Bar(x=hk['Tanggal Servis'].astype(str),y=hk['Jumlah Servis'],
                                name='Hist 2025',marker_color='#DAEAF8',marker_line_width=0,opacity=0.8))
            fv.add_trace(go.Scatter(x=pd.concat([fds,fds[::-1]]),
                                    y=pd.concat([fck['optimistic'],fck['pessimistic'][::-1]]),
                                    fill='toself',fillcolor='rgba(204,0,0,0.05)',
                                    line=dict(color='rgba(0,0,0,0)'),name='Rentang',showlegend=False))
            fv.add_trace(go.Scatter(x=fds,y=fck['forecast'],mode='lines',name='Prakiraan',
                                    line=dict(color=clr,width=1.5)))
            fv.add_trace(go.Scatter(x=ds13,y=act,mode='lines+markers',name='Aktual 2026',
                                    line=dict(color='#CC0000',width=1.8),
                                    marker=dict(size=3,color='#CC0000')))
            lb=LEB.get(2026)
            if lb:
                fv.add_shape(type='line',x0=lb,x1=lb,y0=0,y1=1,yref='paper',
                             line=dict(color='#00AA00',width=0.8,dash='dash'))
            fv.update_layout(**pb(175,
                title=dict(text=f"{dl(kec)} · MAPE {mv:.1f}%",
                           font=dict(size=9,color='#333'),x=0.5,xanchor='center'),
                margin=dict(l=6,r=6,t=22,b=55),
                xaxis=dict(tickformat='%b %Y',showgrid=True,gridcolor='#f5f5f5',
                           linecolor='#e8e8e8',tickfont=dict(size=7,color='#aaa')),
                yaxis=dict(showgrid=True,gridcolor='#f5f5f5',
                           tickfont=dict(size=7,color='#aaa')),
                legend=dict(orientation='h',y=-0.42,x=0.5,xanchor='center',
                            font=dict(size=7),bgcolor='rgba(255,255,255,0.9)')))
            with col:
                st.plotly_chart(fv,use_container_width=True,config={'displayModeBar':False})

    with rc:
        st.markdown('<div class="sh">Metrik Validasi</div>', unsafe_allow_html=True)
        disp=dv[['District','MAPE (%)','RMSE','MAE','Bias']].sort_values('MAPE (%)')
        st.dataframe(
            disp.style
                .highlight_min(subset=['MAPE (%)'],color='#d4edda')
                .highlight_max(subset=['MAPE (%)'],color='#f8d7da')
                .format({'MAPE (%)':'{:.1f}%','RMSE':'{:,}','MAE':'{:,}','Bias':'{:+,}'}),
            use_container_width=True,hide_index=True,height=220)

        st.markdown('<div class="sh" style="margin-top:10px">Interpretasi</div>', unsafe_allow_html=True)
        for clr,bg,title,desc in [
            ("#16a34a","#f0fdf4","✅ Performa Baik","District A, C, E — MAPE < 12%."),
            ("#d97706","#fffbeb","⚠️ Perlu Perhatian","District B & D melebihi target MAPE 15%."),
            ("#CC0000","#fff5f5","🔴 Efek Lebaran","Penurunan permintaan Lebaran di luar rentang pelatihan."),
        ]:
            st.markdown(f"""
            <div class="ic" style="background:{bg};border-color:{clr}">
              <p class="ic-t" style="color:{clr}">{title}</p>
              <p class="ic-b">{desc}</p>
            </div>""",unsafe_allow_html=True)
