import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e35;
}
[data-testid="stSidebar"] * {
    color: #c8c8e0 !important;
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #12122a 50%, #0a1628 100%);
    border: 1px solid #1e2a4a;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,180,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #00b4ff, #00ffc8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #5a6a8a;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 20px 0;
}
.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e35;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00b4ff, #00ffc8);
    border-radius: 3px 0 0 3px;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #5a6a8a;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8e8f0;
    letter-spacing: -0.5px;
}
.metric-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    margin-top: 4px;
}
.positive { color: #00ffc8; }
.negative { color: #ff4f6d; }

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 18px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #1e1e35;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8e8f0;
    letter-spacing: -0.3px;
}
.section-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    background: #1a2a4a;
    color: #00b4ff;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Prediction result box */
.pred-box {
    background: linear-gradient(135deg, #0a1628, #0f1f3d);
    border: 1px solid #1a3a6a;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin: 20px 0;
}
.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #5a8abf;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.pred-price {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00b4ff, #00ffc8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px;
}
.pred-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #5a6a8a;
    margin-top: 8px;
}

/* Info tag */
.info-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    background: #1a1a2e;
    color: #00ffc8;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 4px;
    letter-spacing: 0.5px;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #7a8aaa !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
.stButton > button {
    background: linear-gradient(90deg, #00b4ff, #00ffc8) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

div[data-testid="stMetric"] {
    background: #0f0f1a;
    border: 1px solid #1e1e35;
    border-radius: 10px;
    padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0f0f1a',
    'axes.facecolor':    '#0f0f1a',
    'axes.edgecolor':    '#1e1e35',
    'axes.labelcolor':   '#8a8aaa',
    'axes.titlecolor':   '#e8e8f0',
    'xtick.color':       '#5a6a8a',
    'ytick.color':       '#5a6a8a',
    'grid.color':        '#1a1a2e',
    'grid.linewidth':    0.6,
    'text.color':        '#e8e8f0',
    'legend.facecolor':  '#0f0f1a',
    'legend.edgecolor':  '#1e1e35',
    'figure.dpi':        130,
    'font.family':       'monospace',
})

ACCENT   = '#00b4ff'
GREEN    = '#00ffc8'
RED      = '#ff4f6d'
PURPLE   = '#a78bfa'
ORANGE   = '#fb923c'
COLORS   = [ACCENT, GREEN, RED, PURPLE]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    POPULAR = {
        'Apple (AAPL)': 'AAPL',
        'Google (GOOG)': 'GOOG',
        'Microsoft (MSFT)': 'MSFT',
        'Amazon (AMZN)': 'AMZN',
        'Tesla (TSLA)': 'TSLA',
        'NVIDIA (NVDA)': 'NVDA',
        'Meta (META)': 'META',
        'Netflix (NFLX)': 'NFLX',
    }

    st.markdown("**PRIMARY STOCK**")
    selected_name = st.selectbox("Select stock", list(POPULAR.keys()), label_visibility='collapsed')
    primary_ticker = POPULAR[selected_name]

    custom = st.text_input("Or type any ticker (e.g. RELIANCE.NS)", "").upper().strip()
    if custom:
        primary_ticker = custom

    st.markdown("---")
    st.markdown("**DATE RANGE**")
    period_map = {'3 Months': 90, '6 Months': 180, '1 Year': 365, '2 Years': 730}
    period_label = st.radio("Analysis period", list(period_map.keys()), index=2, label_visibility='collapsed')
    days = period_map[period_label]

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days)

    st.markdown("---")
    st.markdown("**COMPARE WITH**")
    compare_options = [k for k in POPULAR if POPULAR[k] != primary_ticker]
    compare_names   = st.multiselect("Add stocks", compare_options, default=[], label_visibility='collapsed')
    compare_tickers = [POPULAR[n] for n in compare_names]
    all_tickers     = [primary_ticker] + compare_tickers

    st.markdown("---")
    st.markdown("**LSTM PREDICTION**")
    run_lstm = st.checkbox("Enable price prediction", value=True)
    lookback = st.slider("Lookback window (days)", 30, 90, 60)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:monospace;font-size:0.68rem;color:#3a4a6a;line-height:1.8'>
    📌 Data via Yahoo Finance<br>
    🤖 LSTM Neural Network<br>
    👤 Amrit Jha — B.Tech 3rd Year
    </div>
    """, unsafe_allow_html=True)

    analyse_btn = st.button("🚀 Run Analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <div class="hero-title">StockSense AI 📈</div>
    <div class="hero-sub">Real-time Analysis · Moving Averages · LSTM Prediction · Risk Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
if not analyse_btn:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#3a4a6a'>
        <div style='font-size:3rem'>📊</div>
        <div style='font-family:monospace;font-size:0.85rem;margin-top:12px;letter-spacing:2px'>
            SELECT A STOCK AND CLICK RUN ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Data Fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_stock(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

@st.cache_data(ttl=300)
def fetch_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info
    except:
        return {}

with st.spinner(f"Fetching live data for {', '.join(all_tickers)}..."):
    stock_data = {}
    failed = []
    for t in all_tickers:
        df = fetch_stock(t, start_date, end_date)
        if df.empty:
            failed.append(t)
        else:
            stock_data[t] = df

if failed:
    st.error(f"❌ Could not fetch: {', '.join(failed)}. Check ticker symbol.")

if primary_ticker not in stock_data:
    st.stop()

df_main = stock_data[primary_ticker]

# ── Company Info ──────────────────────────────────────────────────────────────
info = fetch_info(primary_ticker)
company_name = info.get('longName', primary_ticker)
sector       = info.get('sector', 'N/A')
industry     = info.get('industry', 'N/A')
market_cap   = info.get('marketCap', None)

st.markdown(f"""
<div style='margin-bottom:20px'>
    <span style='font-size:1.5rem;font-weight:800;color:#e8e8f0'>{company_name}</span>
    <span style='font-family:monospace;font-size:0.75rem;color:#5a6a8a;margin-left:14px'>
        {primary_ticker} · {sector} · {industry}
    </span>
</div>
""", unsafe_allow_html=True)

# ── Key Metrics ───────────────────────────────────────────────────────────────
latest      = df_main['Close'].iloc[-1]
prev        = df_main['Close'].iloc[-2]
change      = latest - prev
change_pct  = (change / prev) * 100
high_52w    = df_main['Close'].max()
low_52w     = df_main['Close'].min()
avg_vol     = df_main['Volume'].mean()
volatility  = df_main['Close'].pct_change().std() * np.sqrt(252) * 100

col1, col2, col3, col4 = st.columns(4)
delta_color = "positive" if change >= 0 else "negative"
delta_sign  = "▲" if change >= 0 else "▼"

with col1:
    st.metric("Current Price", f"${latest:.2f}", f"{delta_sign} {abs(change_pct):.2f}% today")
with col2:
    st.metric("52W High", f"${high_52w:.2f}")
with col3:
    st.metric("52W Low", f"${low_52w:.2f}")
with col4:
    st.metric("Annualised Volatility", f"{volatility:.1f}%")

if market_cap:
    st.markdown(f"""
    <span class='info-tag'>Market Cap: ${market_cap/1e9:.1f}B</span>
    <span class='info-tag'>Avg Volume: {avg_vol/1e6:.1f}M</span>
    <span class='info-tag'>Period: {start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}</span>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PRICE & VOLUME
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-header'>
    <span class='section-title'>📉 Price & Volume History</span>
    <span class='section-badge'>EDA</span>
</div>""", unsafe_allow_html=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#0f0f1a')

# Price plot
for i, t in enumerate(all_tickers):
    if t in stock_data:
        color = COLORS[i % len(COLORS)]
        ax1.plot(stock_data[t].index, stock_data[t]['Close'],
                 color=color, linewidth=1.8, label=t, alpha=0.9)
        if t == primary_ticker:
            ax1.fill_between(stock_data[t].index,
                             stock_data[t]['Close'],
                             stock_data[t]['Close'].min(),
                             alpha=0.06, color=color)

ax1.set_ylabel('Close Price (USD)', fontsize=9)
ax1.set_title(f'Closing Price — {", ".join(all_tickers)}', fontsize=11, fontweight='bold', pad=12)
ax1.legend(fontsize=9, framealpha=0.4)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax1.grid(True, alpha=0.3)

# Volume bar
ax2.bar(df_main.index, df_main['Volume'] / 1e6,
        color=ACCENT, alpha=0.5, width=1.2)
ax2.set_ylabel('Vol (M)', fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax2.grid(True, alpha=0.2)

plt.tight_layout(h_pad=1.5)
st.pyplot(fig)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MOVING AVERAGES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-header'>
    <span class='section-title'>📊 Moving Averages</span>
    <span class='section-badge'>TREND</span>
</div>""", unsafe_allow_html=True)

MA_WINDOWS = [10, 20, 50]
MA_COLORS  = ['#fb923c', '#a78bfa', RED]

for w in MA_WINDOWS:
    df_main[f'MA_{w}'] = df_main['Close'].rolling(w).mean()

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor('#0f0f1a')

ax.plot(df_main.index, df_main['Close'], color=ACCENT, linewidth=1.8,
        label='Close Price', alpha=0.95)
for w, c in zip(MA_WINDOWS, MA_COLORS):
    ax.plot(df_main.index, df_main[f'MA_{w}'],
            color=c, linewidth=1.3, linestyle='--', label=f'{w}-Day MA', alpha=0.85)

ax.set_title(f'{primary_ticker} — Moving Averages (10 / 20 / 50 Day)', fontsize=11, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=9)
ax.legend(fontsize=9, framealpha=0.4)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# MA crossover signal
ma10 = df_main['MA_10'].iloc[-1]
ma50 = df_main['MA_50'].iloc[-1]
if ma10 > ma50:
    st.success(f"📈 **Bullish Signal** — 10-day MA (${ma10:.2f}) is above 50-day MA (${ma50:.2f}). Short-term upward momentum.")
else:
    st.warning(f"📉 **Bearish Signal** — 10-day MA (${ma10:.2f}) is below 50-day MA (${ma50:.2f}). Possible downward pressure.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DAILY RETURNS & RISK
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-header'>
    <span class='section-title'>⚡ Daily Returns & Risk Analysis</span>
    <span class='section-badge'>RISK</span>
</div>""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)

df_main['Daily Return'] = df_main['Close'].pct_change()

with col_a:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0f0f1a')
    dr = df_main['Daily Return'].dropna() * 100
    ax.plot(dr.index, dr, color=ACCENT, linewidth=0.8, alpha=0.8)
    ax.axhline(0, color='#3a3a5a', linewidth=1, linestyle='--')
    ax.fill_between(dr.index, dr, 0, where=(dr >= 0), alpha=0.25, color=GREEN)
    ax.fill_between(dr.index, dr, 0, where=(dr < 0),  alpha=0.25, color=RED)
    ax.set_title('Daily Return (%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Return (%)', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_b:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0f0f1a')
    dr = df_main['Daily Return'].dropna() * 100
    ax.hist(dr, bins=50, color=ACCENT, alpha=0.7, edgecolor='#0f0f1a', linewidth=0.4)
    ax.axvline(dr.mean(), color=GREEN, linestyle='--', linewidth=1.4,
               label=f'Mean: {dr.mean():.3f}%')
    ax.axvline(0, color='#5a5a8a', linestyle='--', linewidth=0.8)
    ax.set_title('Return Distribution', fontsize=10, fontweight='bold')
    ax.set_xlabel('Daily Return (%)', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.legend(fontsize=8, framealpha=0.4)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Risk stats
dr = df_main['Daily Return'].dropna()
r1, r2, r3, r4 = st.columns(4)
r1.metric("Mean Daily Return", f"{dr.mean()*100:.3f}%")
r2.metric("Daily Std Dev",     f"{dr.std()*100:.3f}%")
r3.metric("Best Day",          f"{dr.max()*100:.2f}%")
r4.metric("Worst Day",         f"{dr.min()*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORRELATION (if comparing)
# ═══════════════════════════════════════════════════════════════════════════════
if len(all_tickers) > 1:
    st.markdown("""
    <div class='section-header'>
        <span class='section-title'>🔗 Correlation Analysis</span>
        <span class='section-badge'>COMPARE</span>
    </div>""", unsafe_allow_html=True)

    closing_df = pd.DataFrame({t: stock_data[t]['Close'] for t in stock_data})
    returns_df = closing_df.pct_change().dropna()

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0f0f1a')
        mask = np.zeros_like(returns_df.corr(), dtype=bool)
        sns.heatmap(returns_df.corr(), annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True,
                    linewidths=1, linecolor='#0a0a0f',
                    ax=ax, cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 10, 'weight': 'bold'})
        ax.set_title('Returns Correlation', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_h2:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0f0f1a')

        mean_r = returns_df.mean() * 100
        std_r  = returns_df.std()  * 100
        for i, t in enumerate(returns_df.columns):
            c = COLORS[i % len(COLORS)]
            ax.scatter(std_r[t], mean_r[t], s=200, color=c,
                       edgecolors='white', linewidth=0.8, zorder=5)
            ax.annotate(f'  {t}', (std_r[t], mean_r[t]),
                        fontsize=10, fontweight='bold', color=c)

        ax.axhline(0, color='#3a3a5a', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Risk (Std Dev of Daily Return %)', fontsize=8)
        ax.set_ylabel('Mean Daily Return (%)', fontsize=8)
        ax.set_title('Risk vs. Return', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LSTM PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if run_lstm:
    st.markdown("""
    <div class='section-header'>
        <span class='section-title'>🤖 LSTM Price Prediction</span>
        <span class='section-badge'>DEEP LEARNING</span>
    </div>""", unsafe_allow_html=True)

    st.info(f"Training LSTM on 3 years of **{primary_ticker}** data with a {lookback}-day lookback window. This may take 1–2 minutes.")

    @st.cache_data(ttl=3600)
    def run_lstm_model(ticker, lookback):
        try:
            from keras.models import Sequential
            from keras.layers import Dense, LSTM, Dropout
            from keras.callbacks import EarlyStopping

            lstm_end   = datetime.now()
            lstm_start = lstm_end - timedelta(days=3 * 365)

            df_lstm = yf.download(ticker, start=lstm_start, end=lstm_end,
                                  progress=False, auto_adjust=True)
            df_lstm.columns = [c[0] if isinstance(c, tuple) else c for c in df_lstm.columns]

            dataset = df_lstm[['Close']].values
            if len(dataset) < lookback + 50:
                return None

            scaler     = MinMaxScaler(feature_range=(0, 1))
            scaled     = scaler.fit_transform(dataset)
            train_len  = int(len(scaled) * 0.90)

            def make_seqs(data, lb):
                X, y = [], []
                for i in range(lb, len(data)):
                    X.append(data[i-lb:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)

            X_tr, y_tr = make_seqs(scaled[:train_len], lookback)
            X_te, y_te = make_seqs(scaled[train_len - lookback:], lookback)

            X_tr = X_tr.reshape(*X_tr.shape, 1)
            X_te = X_te.reshape(*X_te.shape, 1)

            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_tr, y_tr, batch_size=32, epochs=30,
                      validation_split=0.1,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=6,
                                               restore_best_weights=True, verbose=0)],
                      verbose=0)

            preds = scaler.inverse_transform(model.predict(X_te, verbose=0))
            y_actual = scaler.inverse_transform(y_te.reshape(-1, 1))

            # Predict NEXT day
            last_seq   = scaled[-lookback:].reshape(1, lookback, 1)
            next_price = scaler.inverse_transform(model.predict(last_seq, verbose=0))[0][0]

            rmse = np.sqrt(mean_squared_error(y_actual, preds))
            mae  = mean_absolute_error(y_actual, preds)
            mape = np.mean(np.abs((y_actual - preds) / y_actual)) * 100

            return {
                'preds': preds.flatten(),
                'actual': y_actual.flatten(),
                'next_price': next_price,
                'dates': df_lstm.index[train_len:],
                'train_close': df_lstm['Close'].values[:train_len],
                'train_dates': df_lstm.index[:train_len],
                'rmse': rmse, 'mae': mae, 'mape': mape,
                'current_price': dataset[-1][0]
            }
        except Exception as e:
            return {'error': str(e)}

    with st.spinner("🧠 Training neural network..."):
        result = run_lstm_model(primary_ticker, lookback)

    if result is None:
        st.error("Not enough historical data for LSTM. Try a ticker with more history.")
    elif 'error' in result:
        st.error(f"LSTM error: {result['error']}")
    else:
        # Next day prediction box
        next_p   = result['next_price']
        curr_p   = result['current_price']
        diff     = next_p - curr_p
        diff_pct = (diff / curr_p) * 100
        direction = "▲" if diff >= 0 else "▼"
        dir_color = "#00ffc8" if diff >= 0 else "#ff4f6d"

        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%d %b %Y')
        st.markdown(f"""
        <div class='pred-box'>
            <div class='pred-label'>Predicted Closing Price — {tomorrow}</div>
            <div class='pred-price'>${next_p:.2f}</div>
            <div class='pred-meta' style='color:{dir_color}'>
                {direction} ${abs(diff):.2f} ({abs(diff_pct):.2f}%) from current ${curr_p:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"${result['rmse']:.2f}", help="Root Mean Squared Error")
        m2.metric("MAE",  f"${result['mae']:.2f}",  help="Mean Absolute Error")
        m3.metric("MAPE", f"{result['mape']:.2f}%", help="Mean Absolute Percentage Error")

        # Prediction chart
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor('#0f0f1a')

        ax.plot(result['train_dates'], result['train_close'],
                color='#3a4a6a', linewidth=1.2, label='Training Data', alpha=0.7)
        ax.plot(result['dates'], result['actual'],
                color=GREEN, linewidth=1.8, label='Actual Price')
        ax.plot(result['dates'], result['preds'],
                color=RED, linewidth=1.8, linestyle='--', label='Predicted Price')

        # Mark next-day prediction
        ax.scatter([result['dates'][-1] + timedelta(days=1)], [next_p],
                   color=ORANGE, s=120, zorder=6, label=f'Next Day: ${next_p:.2f}')

        ax.set_title(f'{primary_ticker} — LSTM Predicted vs Actual (Test Set)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Close Price (USD)', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.legend(fontsize=9, framealpha=0.4)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Disclaimer
        st.caption("⚠️ This prediction is generated by an ML model for educational purposes only. It is NOT financial advice. Always consult a qualified advisor before making investment decisions.")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:monospace;font-size:0.7rem;color:#2a3a5a;padding:16px'>
    StockSense AI · Built by Amrit Jha · B.Tech 3rd Year · Data via Yahoo Finance
</div>
""", unsafe_allow_html=True)
