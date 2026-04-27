import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AHASS Demand Forecasting",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #FFFFFF; color: #0D0D0D; }
    .stApp { background-color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #CC0000 !important; border-right: 3px solid #0D0D0D; }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stSelectbox > div > div { background-color: #FFFFFF !important; color: #0D0D0D !important; border: 2px solid #0D0D0D; }
    .main-header { background-color: #CC0000; padding: 2rem 2.5rem; margin: -1rem -1rem 2rem -1rem; border-bottom: 4px solid #0D0D0D; }
    .main-header h1 { font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem; color: #FFFFFF; letter-spacing: 3px; margin: 0; line-height: 1; }
    .main-header p { color: #FFD0D0; font-size: 1rem; margin: 0.5rem 0 0 0; }
    .section-header { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; letter-spacing: 2px; color: #0D0D0D; border-bottom: 3px solid #CC0000; padding-bottom: 0.5rem; margin: 2rem 0 1rem 0; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 3px solid #CC0000; gap: 0; }
    .stTabs [data-baseweb="tab"] { background-color: #FFFFFF; color: #0D0D0D; font-weight: 600; letter-spacing: 1px; border: 2px solid #0D0D0D; border-bottom: none; padding: 0.5rem 1.5rem; }
    .stTabs [aria-selected="true"] { background-color: #CC0000 !important; color: #FFFFFF !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = "streamlit_assets"
    with open(f"{base}/lgb_models.pkl", "rb") as f:
        lgb_models = pickle.load(f)
    with open(f"{base}/forecast_results.pkl", "rb") as f:
        forecast_results = pickle.load(f)
    with open(f"{base}/shap_results.pkl", "rb") as f:
        shap_results = pickle.load(f)
    with open(f"{base}/config.pkl", "rb") as f:
        config = pickle.load(f)
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

# Shared layout
LEGEND = dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5,
              bgcolor='rgba(255,255,255,0.95)', bordercolor='#0D0D0D', borderwidth=1,
              font=dict(size=10, color='#0D0D0D'))

def base_layout(**kwargs):
    d = dict(plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
             font=dict(color='#0D0D0D', family='IBM Plex Sans'),
             xaxis=dict(showgrid=True, gridcolor='#F0F0F0', linecolor='#0D0D0D',
                        tickfont=dict(color='#0D0D0D')),
             yaxis=dict(showgrid=True, gridcolor='#F0F0F0', linecolor='#0D0D0D',
                        tickfont=dict(color='#0D0D0D')),
             margin=dict(l=20, r=20, t=30, b=110),
             legend=LEGEND)
    d.update(kwargs)
    return d

# ── Make forecast chart ───────────────────────────────────────────────────────
def make_forecast_chart(kec, sel_idx=None):
    fc    = forecast_results[kec]
    hist  = dfs_smooth[dfs_smooth['Kecamatan Bengkel'] == kec].tail(26)
    color = KEC_COLOR_MAP[kec]
    fc_ds = fc['ds'].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist['Tanggal Servis'].astype(str), y=hist['Jumlah Servis'],
                         name='Historical 2025', marker_color='#B5D4F4', opacity=0.7))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_ds, fc_ds[::-1]]),
        y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
        fill='toself', fillcolor='rgba(204,0,0,0.1)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'))
    fig.add_trace(go.Scatter(x=fc_ds, y=fc['pessimistic'], mode='lines', name='Pessimistic -16%', line=dict(color='#AAAAAA', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=fc_ds, y=fc['optimistic'],  mode='lines', name='Optimistic +18%',  line=dict(color='#3B6D11', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=fc_ds, y=fc['forecast'],    mode='lines', name='Forecast (LightGBM)', line=dict(color=color, width=3)))

    lb26 = LEBARAN_STR.get(2026)
    if lb26:
        fig.add_shape(type='line', x0=lb26, x1=lb26, y0=0, y1=1, yref='paper',
                      line=dict(color='#00AA00', width=2, dash='dash'))
    if sel_idx is not None:
        sel_ds = str(fc['ds'].iloc[sel_idx])
        fig.add_shape(type='line', x0=sel_ds, x1=sel_ds, y0=0, y1=1, yref='paper',
                      line=dict(color='#CC0000', width=2, dash='dot'))

    fig.update_layout(**base_layout(height=480, xaxis=dict(tickformat='%b %Y',
                      showgrid=True, gridcolor='#F0F0F0', linecolor='#0D0D0D',
                      tickfont=dict(color='#0D0D0D'))))
    return fig

# ── Make SHAP bar ─────────────────────────────────────────────────────────────
def make_shap_bar(kec):
    sr        = shap_results[kec]
    mean_shap = np.abs(sr['shap_values']).mean(axis=0)
    df = pd.DataFrame({'Feature': sr['feat_cols'], 'Importance': mean_shap}).sort_values('Importance')
    fig = go.Figure(go.Bar(x=df['Importance'], y=df['Feature'], orientation='h',
                           marker_color='#CC0000', marker_line_color='#0D0D0D', marker_line_width=1))
    fig.update_layout(**base_layout(height=420,
                      xaxis=dict(title='Mean |SHAP Value|', showgrid=True,
                                 gridcolor='#F0F0F0', linecolor='#0D0D0D',
                                 tickfont=dict(color='#0D0D0D'))))
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏍️ AHASS DEMAND FORECASTING</h1>
    <p>Weekly Service Demand Forecast — Top 5 Kecamatan Jakarta · LightGBM + SHAP XAI</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CONTROLS")
    st.markdown("---")
    selected_kec        = st.selectbox("📍 Kecamatan", options=TOP5, index=0)
    fc_data             = forecast_results[selected_kec]
    week_options        = [f"Week {i+1} — {d.strftime('%d %b %Y')}" for i, d in enumerate(fc_data['ds'])]
    selected_week_label = st.selectbox("📅 Forecast Week", options=week_options, index=0)
    selected_week_idx   = week_options.index(selected_week_label)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("**Best Model:** LightGBM")
    st.markdown(f"**Features:** {len(FEAT_COLS)} lag + calendar")
    st.markdown("**Training:** 2022–2024")
    st.markdown("**Test:** 2025")
    st.markdown("---")
    st.markdown("### 🎯 2025 Test MAPE")
    st.markdown(f"**{selected_kec}:** `{lgb_models[selected_kec]['mape']*100:.1f}%`")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Thesis project — Wahana Artha · 2025")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 FORECAST", "🔍 XAI EXPLANATION", "📊 MODEL COMPARISON", "✅ 2026 VALIDATION"])

# ═══ TAB 1 ═══════════════════════════════════════════════════════════════════
with tab1:
    fc_row = fc_data.iloc[selected_week_idx]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📍 Kecamatan", selected_kec)
    with c2: st.metric("🔮 Base Forecast", f"{fc_row['forecast']:,.0f}", help="services/week")
    with c3: st.metric("📈 Optimistic +18%", f"{fc_row['optimistic']:,.0f}")
    with c4: st.metric("📉 Pessimistic -16%", f"{fc_row['pessimistic']:,.0f}")

    st.markdown('<div class="section-header">52-WEEK FORECAST CHART</div>', unsafe_allow_html=True)
    st.plotly_chart(make_forecast_chart(selected_kec, selected_week_idx), use_container_width=True)

    st.markdown('<div class="section-header">ALL KECAMATAN OVERVIEW</div>', unsafe_allow_html=True)
    cols = st.columns(len(TOP5))
    for i, (kec, color) in enumerate(zip(TOP5, KEC_COLORS)):
        fc_kec = forecast_results[kec]
        with cols[i]:
            st.metric(kec, f"{fc_kec['forecast'].mean():,.0f}", delta=f"annual={fc_kec['forecast'].sum():,.0f}")
            st.caption(f"Test MAPE={lgb_models[kec]['mape']*100:.1f}%")

# ═══ TAB 2 ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">XAI — WHY THIS FORECAST?</div>', unsafe_allow_html=True)
    st.markdown("SHAP (SHapley Additive exPlanations) explains which features drove this prediction and by how much.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        fc_row_xai = forecast_results[selected_kec].iloc[selected_week_idx]
        st.markdown(f"### 🔮 {fc_row_xai['forecast']:,.0f} services/week")
        st.caption(f"{fc_row_xai['ds'].strftime('%d %b %Y')} · {selected_kec}")

        sr      = shap_results[selected_kec]
        idx     = min(selected_week_idx, len(sr['shap_values']) - 1)
        sv      = sr['shap_values'][idx]
        xv      = sr['X_test'][idx]
        shap_df = pd.DataFrame({'feature': sr['feat_cols'], 'shap': sv, 'value': xv})
        shap_df['abs_shap'] = shap_df['shap'].abs()
        shap_df = shap_df.sort_values('abs_shap', ascending=False).head(5)
        max_abs = shap_df['abs_shap'].max()

        st.markdown("#### 🔍 Top 5 Driving Features")
        for _, row in shap_df.iterrows():
            direction = "↑" if row['shap'] > 0 else "↓"
            clr       = "#CC0000" if row['shap'] > 0 else "#0066CC"
            bar_w     = int((row['abs_shap'] / max_abs) * 100)
            ca, cb, cc = st.columns([1, 3, 4])
            with ca: st.markdown(f"<span style='color:{clr};font-size:1.3rem;font-weight:700'>{direction}</span>", unsafe_allow_html=True)
            with cb: st.markdown(f"**{row['feature']}**  \n`val={row['value']:.1f}`")
            with cc:
                st.progress(bar_w)
                st.caption(f"impact: {row['shap']:+.1f}")

        st.markdown("""<div style="background:#F5F5F5;padding:1rem;border-left:4px solid #CC0000;
            font-size:0.85rem;color:#0D0D0D;margin-top:1rem">
            <b>How to read:</b> ↑ Red = pushes forecast UP · ↓ Blue = pushes forecast DOWN · Bar = magnitude
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"#### Global Feature Importance — {selected_kec}")
        st.plotly_chart(make_shap_bar(selected_kec), use_container_width=True)

    st.markdown('<div class="section-header">SHAP — ALL KECAMATAN</div>', unsafe_allow_html=True)
    fig_all = go.Figure()
    for kec, color in zip(TOP5, KEC_COLORS):
        sr        = shap_results[kec]
        mean_shap = np.abs(sr['shap_values']).mean(axis=0)
        fig_all.add_trace(go.Bar(name=kec, x=sr['feat_cols'], y=mean_shap,
                                 marker_color=color, marker_line_color='#0D0D0D', marker_line_width=0.5))
    fig_all.update_layout(**base_layout(barmode='group', height=400,
                           xaxis=dict(tickangle=-45, showgrid=True, gridcolor='#F0F0F0',
                                      linecolor='#0D0D0D', tickfont=dict(color='#0D0D0D')),
                           yaxis=dict(title='Mean |SHAP Value|', showgrid=True,
                                      gridcolor='#F0F0F0', linecolor='#0D0D0D',
                                      tickfont=dict(color='#0D0D0D'))))
    st.plotly_chart(fig_all, use_container_width=True)

# ═══ TAB 3 ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">MODEL COMPARISON — TEST 2025</div>', unsafe_allow_html=True)

    avg_rows = []
    for model_name in df_comparison['Model'].unique():
        sub = df_comparison[df_comparison['Model'] == model_name]
        avg_rows.append({'Model': model_name, 'Avg MAPE (%)': round(sub['MAPE (%)'].mean(), 1),
                         'Avg RMSE': int(sub['RMSE'].mean()), 'Avg MAE': int(sub['MAE'].mean()),
                         'Avg R²': round(sub['R2'].mean(), 3)})
    df_avg = pd.DataFrame(avg_rows).sort_values('Avg MAPE (%)')

    st.markdown("#### Average Performance Across All Kecamatan")
    st.dataframe(df_avg.style.highlight_min(subset=['Avg MAPE (%)'], color='#FFD0D0'), use_container_width=True, height=180)

    st.markdown("#### MAPE by Model and Kecamatan")
    model_colors = {'XGBoost': '#E6A817', 'LightGBM': '#CC0000', 'Stacking': '#639922'}
    fig_comp = go.Figure()
    for model_name in df_comparison['Model'].unique():
        sub = df_comparison[df_comparison['Model'] == model_name]
        fig_comp.add_trace(go.Bar(name=model_name, x=sub['Kecamatan'], y=sub['MAPE (%)'],
                                  marker_color=model_colors.get(model_name, '#999'),
                                  marker_line_color='#0D0D0D', marker_line_width=0.5))
    fig_comp.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                       line=dict(color='#CC0000', width=1.5, dash='dash'))
    fig_comp.update_layout(**base_layout(barmode='group', height=400,
                            yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F0F0F0',
                                       linecolor='#0D0D0D', tickfont=dict(color='#0D0D0D'))))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Full Results Table")
    st.dataframe(df_comparison.sort_values(['Kecamatan', 'MAPE (%)']), use_container_width=True, height=350)

# ═══ TAB 4 ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">2026 OUT-OF-SAMPLE VALIDATION</div>', unsafe_allow_html=True)
    st.info("📋 Validation against actual 2026 data (Jan–Mar). Missing weeks (Feb 9–23) filled with mean. Lebaran weeks shown separately.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Overall Validation Metrics")
        st.dataframe(df_validation.style.highlight_min(subset=['MAPE (%)'], color='#FFD0D0'), use_container_width=True, height=220)
    with col2:
        st.markdown("#### MAPE by Kecamatan")
        fig_val = go.Figure(go.Bar(
            x=df_validation['Kecamatan'], y=df_validation['MAPE (%)'],
            marker_color=[KEC_COLOR_MAP[k] for k in df_validation['Kecamatan']],
            marker_line_color='#0D0D0D', marker_line_width=1,
            text=df_validation['MAPE (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside', textfont=dict(color='#0D0D0D'),
        ))
        fig_val.add_shape(type='line', x0=0, x1=1, xref='paper', y0=15, y1=15,
                          line=dict(color='#CC0000', width=1.5, dash='dash'))
        fig_val.update_layout(**base_layout(height=280, showlegend=False,
                               yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='#F0F0F0',
                                          linecolor='#0D0D0D', tickfont=dict(color='#0D0D0D'))))
        st.plotly_chart(fig_val, use_container_width=True)

    st.markdown('<div class="section-header">INTERPRETATION</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.success("✅ **Strong Performance**\n\nKECAMATAN_D & E — MAPE below 12% during normal weeks")
    with c2: st.warning("⚠️ **Higher Error**\n\nKECAMATAN_A & C — Demand volatility exceeds historical patterns")
    with c3: st.error("📅 **Lebaran Effect**\n\n29–75% MAPE — Holiday demand drops outside training range")
