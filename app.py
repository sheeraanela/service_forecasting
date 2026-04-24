import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AHASS Demand Forecasting",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Red, White, Black ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #FFFFFF;
        color: #0D0D0D;
    }

    /* Main background */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #CC0000 !important;
        border-right: 3px solid #0D0D0D;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        color: #0D0D0D !important;
        border: 2px solid #0D0D0D;
    }

    /* Header */
    .main-header {
        background-color: #CC0000;
        padding: 2rem 2.5rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 4px solid #0D0D0D;
    }
    .main-header h1 {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.5rem;
        color: #FFFFFF;
        letter-spacing: 3px;
        margin: 0;
        line-height: 1;
    }
    .main-header p {
        color: #FFD0D0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* Metric cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 2px solid #0D0D0D;
        border-left: 6px solid #CC0000;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #666666;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.2rem;
        color: #CC0000;
        line-height: 1;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #888888;
        margin-top: 0.3rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.8rem;
        letter-spacing: 2px;
        color: #0D0D0D;
        border-bottom: 3px solid #CC0000;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }

    /* XAI explanation box */
    .xai-box {
        background-color: #FFF5F5;
        border: 2px solid #CC0000;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .xai-box .xai-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.2rem;
        letter-spacing: 2px;
        color: #CC0000;
        margin-bottom: 1rem;
    }
    .xai-row {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #FFD0D0;
        font-size: 0.9rem;
    }
    .xai-row:last-child { border-bottom: none; }
    .xai-up   { color: #CC0000; font-weight: 700; font-size: 1.1rem; margin-right: 0.5rem; }
    .xai-down { color: #0066CC; font-weight: 700; font-size: 1.1rem; margin-right: 0.5rem; }
    .xai-feat { font-weight: 600; min-width: 130px; }
    .xai-val  { color: #555; margin-left: 0.5rem; }
    .xai-bar-pos { display: inline-block; background: #CC0000; height: 12px; border-radius: 2px; margin-left: 0.5rem; }
    .xai-bar-neg { display: inline-block; background: #0066CC; height: 12px; border-radius: 2px; margin-left: 0.5rem; }

    /* Forecast badge */
    .forecast-badge {
        background-color: #CC0000;
        color: #FFFFFF;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        letter-spacing: 2px;
        padding: 1.5rem 2rem;
        text-align: center;
        border: 3px solid #0D0D0D;
        margin: 1rem 0;
    }

    /* Table styling */
    .stDataFrame { border: 2px solid #0D0D0D; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 3px solid #CC0000;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        color: #0D0D0D;
        font-weight: 600;
        letter-spacing: 1px;
        border: 2px solid #0D0D0D;
        border-bottom: none;
        padding: 0.5rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #CC0000 !important;
        color: #FFFFFF !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    dfs_smooth   = pd.read_csv(f"{base}/dfs_smooth.csv")
    dfs_smooth['Tanggal Servis'] = pd.to_datetime(dfs_smooth['Tanggal Servis'])
    df_comparison = pd.read_csv(f"{base}/model_comparison.csv")
    df_validation = pd.read_csv(f"{base}/validation_metrics.csv")
    return lgb_models, forecast_results, shap_results, config, dfs_smooth, df_comparison, df_validation

lgb_models, forecast_results, shap_results, config, dfs_smooth, df_comparison, df_validation = load_assets()

TOP5        = config['TOP5']
LEBARAN     = config['LEBARAN']
KEC_COLORS  = config['KEC_COLORS']
BEST_MODEL  = config['BEST_MODEL_NAME']
KEC_COLOR_MAP = dict(zip(TOP5, KEC_COLORS))

LEBARAN_DATES = {yr: pd.Timestamp(v[0]) for yr, v in LEBARAN.items()}

# ── Helper functions ──────────────────────────────────────────────────────────
def get_shap_explanation(kec, week_idx):
    """Get top 5 SHAP explanations for a specific forecast week."""
    fc        = forecast_results[kec]
    sr        = shap_results[kec]
    feat_cols = sr['feat_cols']

    # Use last available SHAP row as proxy
    idx       = min(week_idx, len(sr['shap_values']) - 1)
    sv        = sr['shap_values'][idx]
    xv        = sr['X_test'][idx]

    df = pd.DataFrame({
        'feature': feat_cols,
        'shap':    sv,
        'value':   xv,
    })
    df['abs_shap'] = df['shap'].abs()
    df = df.sort_values('abs_shap', ascending=False).head(5)
    return df


def make_forecast_chart(kec, show_actual=False, actual_df=None):
    fc    = forecast_results[kec]
    hist  = dfs_smooth[dfs_smooth['Kecamatan Bengkel'] == kec].tail(26)
    color = KEC_COLOR_MAP[kec]
    lb26  = LEBARAN_DATES.get(2026)

    fig = go.Figure()

    # Historical bars
    fig.add_trace(go.Bar(
        x=hist['Tanggal Servis'], y=hist['Jumlah Servis'],
        name='Historical 2025',
        marker_color='#B5D4F4', opacity=0.7,
        width=5 * 86400000,
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc['ds'], fc['ds'][::-1]]),
        y=pd.concat([fc['optimistic'], fc['pessimistic'][::-1]]),
        fill='toself',
        fillcolor='rgba(204,0,0,0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band',
        showlegend=True,
    ))

    # Pessimistic
    fig.add_trace(go.Scatter(
        x=fc['ds'], y=fc['pessimistic'],
        mode='lines', name='Pessimistic -16%',
        line=dict(color='#AAAAAA', dash='dash', width=1.5),
    ))

    # Optimistic
    fig.add_trace(go.Scatter(
        x=fc['ds'], y=fc['optimistic'],
        mode='lines', name='Optimistic +18%',
        line=dict(color='#3B6D11', dash='dash', width=1.5),
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc['ds'], y=fc['forecast'],
        mode='lines', name=f'Forecast ({BEST_MODEL})',
        line=dict(color=color, width=3),
    ))

    # Lebaran marker
    fig.add_shape(
    type='line',
    x0=str(fc_row['ds']), x1=str(fc_row['ds']),
    y0=0, y1=1,
    yref='paper',
    line=dict(color='#CC0000', width=2, dash='dot'),
    )

    fig.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#0D0D0D', family='IBM Plex Sans'),
        legend=dict(
            orientation='h', yanchor='bottom',
            y=1.02, xanchor='right', x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#0D0D0D', borderwidth=1,
        ),
        xaxis=dict(
            showgrid=True, gridcolor='#F0F0F0',
            tickformat='%b %Y',
            linecolor='#0D0D0D', linewidth=2,
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#F0F0F0',
            title='Services / Week',
            linecolor='#0D0D0D', linewidth=2,
        ),
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def make_shap_bar(kec):
    sr        = shap_results[kec]
    feat_cols = sr['feat_cols']
    mean_shap = np.abs(sr['shap_values']).mean(axis=0)

    df = pd.DataFrame({
        'Feature':    feat_cols,
        'Importance': mean_shap,
    }).sort_values('Importance')

    fig = go.Figure(go.Bar(
        x=df['Importance'], y=df['Feature'],
        orientation='h',
        marker_color='#CC0000',
        marker_line_color='#0D0D0D',
        marker_line_width=1,
    ))
    fig.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#0D0D0D', family='IBM Plex Sans'),
        xaxis=dict(
            title='Mean |SHAP Value|',
            showgrid=True, gridcolor='#F0F0F0',
            linecolor='#0D0D0D',
        ),
        yaxis=dict(linecolor='#0D0D0D'),
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏍️ AHASS DEMAND FORECASTING</h1>
    <p>Weekly Service Demand Forecast — Top 5 Kecamatan Jakarta · Powered by LightGBM + SHAP XAI</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CONTROLS")
    st.markdown("---")

    selected_kec = st.selectbox(
        "📍 Kecamatan",
        options=TOP5,
        index=0,
    )

    fc_data      = forecast_results[selected_kec]
    week_options = [f"Week {i+1} — {d.strftime('%d %b %Y')}"
                    for i, d in enumerate(fc_data['ds'])]
    selected_week_label = st.selectbox(
        "📅 Forecast Week",
        options=week_options,
        index=0,
    )
    selected_week_idx = week_options.index(selected_week_label)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"**Best Model:** {BEST_MODEL}")
    st.markdown(f"**Features:** 17 lag + calendar")
    st.markdown(f"**Training:** 2022–2024")
    st.markdown(f"**Test:** 2025")

    st.markdown("---")
    st.markdown("### 🎯 2025 Test MAPE")
    r = lgb_models[selected_kec]
    st.markdown(f"**{selected_kec}:** `{r['mape']*100:.1f}%`")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Thesis project — Wahana Artha · "
        "Universitas · 2025"
    )


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 FORECAST",
    "🔍 XAI EXPLANATION",
    "📊 MODEL COMPARISON",
    "✅ 2026 VALIDATION",
])


# ═══════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ═══════════════════════════════════════════════════════
with tab1:
    fc_row = fc_data.iloc[selected_week_idx]

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">📍 Kecamatan</div>
            <div class="value" style="font-size:1.5rem">{selected_kec}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">🔮 Base Forecast</div>
            <div class="value">{fc_row['forecast']:,.0f}</div>
            <div class="sub">services / week</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">📈 Optimistic +18%</div>
            <div class="value" style="color:#3B6D11">{fc_row['optimistic']:,.0f}</div>
            <div class="sub">upper bound</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">📉 Pessimistic -16%</div>
            <div class="value" style="color:#555">{fc_row['pessimistic']:,.0f}</div>
            <div class="sub">lower bound</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">52-WEEK FORECAST CHART</div>',
                unsafe_allow_html=True)

    # Highlight selected week
    fig = make_forecast_chart(selected_kec)
    fig.add_vline(
        x=fc_row['ds'],
        line_dash='dot',
        line_color='#CC0000',
        line_width=2,
        annotation_text=f'Selected: {fc_row["ds"].strftime("%d %b %Y")}',
        annotation_font_color='#CC0000',
    )
    st.plotly_chart(fig, use_container_width=True)

    # All kecamatan summary
    st.markdown('<div class="section-header">ALL KECAMATAN OVERVIEW</div>',
                unsafe_allow_html=True)

    cols = st.columns(len(TOP5))
    for i, (kec, color) in enumerate(zip(TOP5, KEC_COLORS)):
        fc_kec    = forecast_results[kec]
        avg_fc    = fc_kec['forecast'].mean()
        annual    = fc_kec['forecast'].sum()
        mape_test = lgb_models[kec]['mape'] * 100
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div class="label">{kec}</div>
                <div class="value" style="color:{color};font-size:1.6rem">{avg_fc:,.0f}</div>
                <div class="sub">avg/week · annual={annual:,.0f}</div>
                <div class="sub">Test MAPE={mape_test:.1f}%</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# TAB 2 — XAI EXPLANATION
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">XAI — WHY THIS FORECAST?</div>',
                unsafe_allow_html=True)
    st.markdown(
        "SHAP (SHapley Additive exPlanations) explains **which features "
        "drove this prediction** and by how much."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Forecast badge
        fc_row_xai = forecast_results[selected_kec].iloc[selected_week_idx]
        st.markdown(f"""
        <div class="forecast-badge">
            {fc_row_xai['forecast']:,.0f}
            <div style="font-size:1rem;letter-spacing:3px;opacity:0.8">
                SERVICES / WEEK · {fc_row_xai['ds'].strftime('%d %b %Y')}
            </div>
        </div>""", unsafe_allow_html=True)

        # SHAP explanations
        shap_df = get_shap_explanation(selected_kec, selected_week_idx)
        max_abs = shap_df['abs_shap'].max()

        html_rows = ""
        for _, row in shap_df.iterrows():
            direction  = "↑" if row['shap'] > 0 else "↓"
            arrow_cls  = "xai-up" if row['shap'] > 0 else "xai-down"
            bar_cls    = "xai-bar-pos" if row['shap'] > 0 else "xai-bar-neg"
            bar_width  = int((row['abs_shap'] / max_abs) * 80)
            html_rows += f"""
            <div class="xai-row">
                <span class="{arrow_cls}">{direction}</span>
                <span class="xai-feat">{row['feature']}</span>
                <span class="xai-val">val={row['value']:.1f}</span>
                <span class="{bar_cls}" style="width:{bar_width}px"></span>
                <span style="margin-left:0.5rem;color:#555;font-size:0.85rem">
                    {row['shap']:+.1f}
                </span>
            </div>"""

        st.markdown(f"""
        <div class="xai-box">
            <div class="xai-title">🔍 TOP 5 DRIVING FEATURES</div>
            {html_rows}
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#F5F5F5;padding:1rem;border-left:4px solid #CC0000;
                    font-size:0.85rem;color:#555;margin-top:1rem">
            <b>How to read:</b> ↑ Red = feature pushes forecast UP · 
            ↓ Blue = feature pushes forecast DOWN · 
            Bar width = magnitude of impact
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Global Feature Importance")
        st.markdown(f"Average SHAP values across all 2025 test weeks — {selected_kec}")
        fig_shap = make_shap_bar(selected_kec)
        st.plotly_chart(fig_shap, use_container_width=True)

    # All kecamatan SHAP comparison
    st.markdown('<div class="section-header">SHAP IMPORTANCE — ALL KECAMATAN</div>',
                unsafe_allow_html=True)

    fig_all = go.Figure()
    for kec, color in zip(TOP5, KEC_COLORS):
        sr        = shap_results[kec]
        feat_cols = sr['feat_cols']
        mean_shap = np.abs(sr['shap_values']).mean(axis=0)
        fig_all.add_trace(go.Bar(
            name=kec,
            x=feat_cols,
            y=mean_shap,
            marker_color=color,
            marker_line_color='#0D0D0D',
            marker_line_width=0.5,
        ))

    fig_all.update_layout(
        barmode='group',
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#0D0D0D', family='IBM Plex Sans'),
        xaxis=dict(tickangle=-45, linecolor='#0D0D0D'),
        yaxis=dict(title='Mean |SHAP Value|', linecolor='#0D0D0D',
                   showgrid=True, gridcolor='#F0F0F0'),
        legend=dict(orientation='h', y=1.05),
        height=400,
        margin=dict(l=20, r=20, t=40, b=80),
    )
    st.plotly_chart(fig_all, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">MODEL COMPARISON — TEST PERIOD 2025</div>',
                unsafe_allow_html=True)

    # Average metrics table
    avg_rows = []
    for model_name in df_comparison['Model'].unique():
        sub = df_comparison[df_comparison['Model'] == model_name]
        avg_rows.append({
            'Model':      model_name,
            'Avg MAPE (%)': round(sub['MAPE (%)'].mean(), 1),
            'Avg RMSE':   int(sub['RMSE'].mean()),
            'Avg MAE':    int(sub['MAE'].mean()),
            'Avg R²':     round(sub['R2'].mean(), 3),
            'Winner 🏆':  '✅' if model_name == BEST_MODEL else '',
        })
    df_avg = pd.DataFrame(avg_rows).sort_values('Avg MAPE (%)')

    st.markdown("#### Average Performance Across All Kecamatan")
    st.dataframe(
        df_avg.style
        .highlight_min(subset=['Avg MAPE (%)'], color='#FFD0D0')
        .highlight_max(subset=['Avg R²'], color='#D0FFD0')
        .format({'Avg MAPE (%)': '{:.1f}%', 'Avg R²': '{:.3f}'}),
        use_container_width=True,
        height=220,
    )

    # MAPE comparison chart
    st.markdown("#### MAPE by Model and Kecamatan")
    fig_comp = go.Figure()
    models   = df_comparison['Model'].unique()
    model_colors = {
        'XGBoost':           '#E6A817',
        'LightGBM':          '#CC0000',
        'Stacking Ensemble': '#639922',
        'SARIMA':            '#D85A30',
        'LSTM':              '#8E44AD',
    }
    for model_name in models:
        sub = df_comparison[df_comparison['Model'] == model_name]
        fig_comp.add_trace(go.Bar(
            name=model_name,
            x=sub['Kecamatan'],
            y=sub['MAPE (%)'],
            marker_color=model_colors.get(model_name, '#999'),
            marker_line_color='#0D0D0D',
            marker_line_width=0.5,
        ))

    fig_comp.add_hline(
        y=15, line_dash='dash',
        line_color='#CC0000', line_width=1.5,
        annotation_text='15% threshold',
        annotation_font_color='#CC0000',
    )
    fig_comp.update_layout(
        barmode='group',
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#0D0D0D', family='IBM Plex Sans'),
        yaxis=dict(
            title='MAPE (%)',
            showgrid=True, gridcolor='#F0F0F0',
            linecolor='#0D0D0D',
        ),
        xaxis=dict(linecolor='#0D0D0D'),
        legend=dict(orientation='h', y=1.05),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Full table
    st.markdown("#### Full Results Table")
    st.dataframe(
        df_comparison.sort_values(['Kecamatan', 'MAPE (%)']),
        use_container_width=True,
        height=350,
    )


# ═══════════════════════════════════════════════════════
# TAB 4 — 2026 VALIDATION
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">2026 OUT-OF-SAMPLE VALIDATION</div>',
                unsafe_allow_html=True)

    st.info(
        "📋 Validation against actual 2026 data (Jan–Mar 2026). "
        "Missing weeks (Feb 9–23) filled with mean. "
        "Lebaran weeks (Mar 23–30) shown separately due to extreme demand volatility."
    )

    # Validation metrics
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Overall Validation Metrics")
        st.dataframe(
            df_validation.style
            .highlight_min(subset=['MAPE (%)'], color='#FFD0D0')
            .format({'MAPE (%)': '{:.1f}%'}),
            use_container_width=True,
            height=220,
        )

    with col2:
        st.markdown("#### MAPE by Kecamatan")
        fig_val = go.Figure(go.Bar(
            x=df_validation['Kecamatan'],
            y=df_validation['MAPE (%)'],
            marker_color=[KEC_COLOR_MAP[k] for k in df_validation['Kecamatan']],
            marker_line_color='#0D0D0D',
            marker_line_width=1,
            text=df_validation['MAPE (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
        ))
        fig_val.add_hline(
            y=15, line_dash='dash',
            line_color='#CC0000', line_width=1.5,
        )
        fig_val.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#0D0D0D', family='IBM Plex Sans'),
            yaxis=dict(
                title='MAPE (%)',
                showgrid=True, gridcolor='#F0F0F0',
            ),
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_val, use_container_width=True)

    # Interpretation
    st.markdown('<div class="section-header">INTERPRETATION</div>',
                unsafe_allow_html=True)

    interp_cols = st.columns(3)
    with interp_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="label">✅ Strong Performance</div>
            <div class="value" style="font-size:1.2rem">KECAMATAN_D & E</div>
            <div class="sub">MAPE below 12% during normal weeks</div>
        </div>""", unsafe_allow_html=True)
    with interp_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="label">⚠️ Higher Error</div>
            <div class="value" style="font-size:1.2rem">KECAMATAN_A & C</div>
            <div class="sub">Demand volatility exceeds historical patterns</div>
        </div>""", unsafe_allow_html=True)
    with interp_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div class="label">📅 Lebaran Effect</div>
            <div class="value" style="font-size:1.2rem">29–75% MAPE</div>
            <div class="sub">Holiday weeks excluded from main evaluation</div>
        </div>""", unsafe_allow_html=True)
