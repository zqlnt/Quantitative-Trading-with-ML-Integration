import streamlit as st
import pandas as pd
import mlflow
from datetime import date
from neural_quant.data.yf_loader import load_yf_data
from neural_quant.strategies.momentum import MovingAverageCrossover
from neural_quant.core.backtest import Backtester

# Page config with professional theme
st.set_page_config(
    page_title="Neural-Quant", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/neural-quant',
        'Report a bug': 'https://github.com/your-repo/neural-quant/issues',
        'About': "Neural-Quant: Advanced Algorithmic Trading Platform"
    }
)

# Custom CSS for professional Google/iOS style
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for professional theming */
    :root {
        --primary-color: #1a73e8;
        --primary-hover: #1557b0;
        --success-color: #34a853;
        --warning-color: #fbbc04;
        --error-color: #ea4335;
        --text-primary: #202124;
        --text-secondary: #5f6368;
        --text-tertiary: #9aa0a6;
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #f1f3f4;
        --border-color: #dadce0;
        --border-hover: #1a73e8;
        --shadow-sm: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
        --shadow-md: 0 1px 2px 0 rgba(60,64,67,0.3), 0 2px 6px 2px rgba(60,64,67,0.15);
        --shadow-lg: 0 2px 4px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
    }
    
    /* Global styles */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1200px;
    }
    
    /* Custom title styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 400;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 400;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-primary);
        border-right: 1px solid var(--border-color);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: var(--text-primary);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        border-radius: 4px;
        padding: 8px 12px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {
        border-color: var(--border-hover);
        box-shadow: 0 0 0 1px var(--border-hover);
        outline: none;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 14px;
        transition: background-color 0.2s ease;
        width: 100%;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-hover);
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: var(--shadow-md);
    }
    
    .metric-title {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 24px;
        font-weight: 400;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    /* Chart containers */
    .chart-container {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: var(--shadow-sm);
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: var(--success-color);
        border: none;
        border-radius: 4px;
        color: white;
        font-weight: 500;
    }
    
    .stInfo {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        color: var(--text-primary);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
    }
    
    /* Spinner styling */
    .stSpinner {
        color: var(--primary-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar toggle button */
    .sidebar-toggle {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1000;
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 12px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        box-shadow: var(--shadow-md);
        transition: all 0.2s ease;
    }
    
    .sidebar-toggle:hover {
        background: var(--primary-hover);
        transform: translateY(-1px);
    }
    
    /* Sidebar state indicator */
    .sidebar-state {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: var(--bg-primary);
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 8px 12px;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        font-weight: 500;
        color: var(--text-primary);
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Card styling */
    .card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: var(--shadow-sm);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-tertiary);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar toggle button
st.markdown("""
<button class="sidebar-toggle" onclick="document.querySelector('[data-testid=stSidebar]').style.display = 'block'; document.querySelector('.sidebar-toggle').style.display = 'none';">
    ☰ Open Sidebar
</button>
<div class="sidebar-state">Sidebar: Open</div>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">Neural-Quant</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Advanced Algorithmic Trading Platform</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: var(--primary-color); font-family: 'Inter', sans-serif; font-weight: 500; font-size: 18px;">Experiment Configuration</h2>
        <p style="color: var(--text-secondary); font-size: 14px; margin: 8px 0 0 0;">Configure your trading strategy parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Market Data</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker Symbol", "AAPL", help="Enter the stock symbol to analyze")
    start = st.date_input("Start Date", value=date(2021,1,1), help="Backtest start date")
    end   = st.date_input("End Date", value=date.today(), help="Backtest end date")
    
    st.markdown('<div class="section-header">Strategy Parameters</div>', unsafe_allow_html=True)
    fast = st.number_input("Fast MA Period", 5, 200, 10, step=1, help="Fast moving average period")
    slow = st.number_input("Slow MA Period", 10, 400, 30, step=1, help="Slow moving average period")
    threshold = st.number_input("Signal Threshold (%)", 0.0, 5.0, 0.2, step=0.1, help="Minimum signal strength to trigger trade")
    
    st.markdown('<div class="section-header">Transaction Costs</div>', unsafe_allow_html=True)
    fee_bps = st.number_input("Commission (bps)", 0, 100, 1, step=1, help="Commission in basis points")
    slip_bps = st.number_input("Slippage (bps)", 0, 100, 2, step=1, help="Slippage in basis points")
    
    st.markdown("---")
    run_btn = st.button("Run Backtest", use_container_width=True)
    
    # Add some info
    st.markdown("""
    <div class="card" style="margin-top: 24px;">
        <h4 style="color: var(--text-primary); margin: 0 0 12px 0; font-size: 14px; font-weight: 500;">Usage Tips</h4>
        <ul style="color: var(--text-secondary); font-size: 13px; margin: 0; padding-left: 16px; line-height: 1.4;">
            <li>Start with default parameters</li>
            <li>Try different MA periods</li>
            <li>Adjust threshold for sensitivity</li>
            <li>Check MLflow for detailed results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# set tracking (adjust if your URI differs)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ma_crossover")

if run_btn:
    with st.spinner("Running..."):
        df = load_yf_data([ticker], str(start), str(end))
        price = df[f"{ticker}_close"].dropna()

        strat = MovingAverageCrossover(ma_fast=fast, ma_slow=slow, threshold=threshold/100.0)
        signals = strat.generate_signals(df)
        # Convert bps to rates (bps / 10000)
        bt = Backtester(commission=fee_bps/10000, slippage=slip_bps/10000)
        results = bt.run_backtest(df, strat)
        equity = results.get('equity_curve', pd.Series())
        trades = results.get('trades', pd.DataFrame())
        metrics = results

        with mlflow.start_run(run_name=f"{ticker}_{fast}-{slow}_{threshold}pct") as run:
            mlflow.log_params({
                "ticker": ticker, "start": str(start), "end": str(end),
                "fast": fast, "slow": slow, "threshold_pct": threshold,
                "fee_bps": fee_bps, "slippage_bps": slip_bps
            })
            for k,v in metrics.items():
                if isinstance(v,(int,float)):
                    mlflow.log_metric(k, float(v))

            # log artifacts
            equity.to_frame("equity").to_csv("equity.csv")
            trades.to_csv("trades.csv", index=False)
            mlflow.log_artifact("equity.csv")
            mlflow.log_artifact("trades.csv")
            run_url = f"{mlflow.get_tracking_uri().rstrip('/')}/#/experiments/{mlflow.get_experiment_by_name('ma_crossover').experiment_id}/runs/{run.info.run_id}"

        # Results header
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: var(--text-primary); font-family: 'Inter', sans-serif; font-weight: 500; font-size: 24px;">Results</h2>
            <p style="color: var(--text-secondary); font-size: 16px; margin: 8px 0 0 0;">Strategy performance analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Return</div>
                <div class="metric-value" style="color: {'var(--primary-color)' if metrics.get('total_return', 0) >= 0 else 'var(--accent-color)'}">
                    {metrics.get('total_return', 0):.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Sharpe Ratio</div>
                <div class="metric-value" style="color: {'var(--primary-color)' if metrics.get('sharpe_ratio', 0) >= 0 else 'var(--accent-color)'}">
                    {metrics.get('sharpe_ratio', 0):.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Max Drawdown</div>
                <div class="metric-value" style="color: var(--accent-color)">
                    {metrics.get('max_drawdown', 0):.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Trades</div>
                <div class="metric-value" style="color: var(--text-primary)">
                    {metrics.get('total_trades', 0)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="chart-container">
                <h3 style="color: var(--text-primary); font-family: 'Inter', sans-serif; margin-bottom: 1rem; font-size: 16px; font-weight: 500;">Equity Curve</h3>
            </div>
            """, unsafe_allow_html=True)
            st.line_chart(equity, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="chart-container">
                <h3 style="color: var(--text-primary); font-family: 'Inter', sans-serif; margin-bottom: 1rem; font-size: 16px; font-weight: 500;">All Metrics</h3>
            </div>
            """, unsafe_allow_html=True)
            # Display metrics in a nicer format
            for key, value in metrics.items():
                if key not in ['equity_curve', 'trades'] and isinstance(value, (int, float)):
                    st.metric(
                        key.replace('_', ' ').title(),
                        f"{value:.2%}" if 'return' in key or 'drawdown' in key else f"{value:.2f}" if isinstance(value, float) else value
                    )
        
        # Trades table
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="color: var(--text-primary); font-family: 'Inter', sans-serif; margin-bottom: 1rem; font-size: 16px; font-weight: 500;">Trade Log</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if not trades.empty:
            st.dataframe(trades, use_container_width=True)
        else:
            st.info("No trades executed during this period.")
        
        # Success message and MLflow link
        st.markdown("""
        <div class="card" style="text-align: center; margin: 2rem 0; background: var(--success-color); color: white;">
            <h3 style="margin: 0 0 8px 0; font-size: 18px; font-weight: 500;">Experiment Complete</h3>
            <p style="margin: 0 0 16px 0; font-size: 14px; opacity: 0.9;">Results have been logged to MLflow for detailed analysis</p>
            <a href="{}" target="_blank" style="color: white; text-decoration: none; font-weight: 500; padding: 8px 16px; background: rgba(255,255,255,0.2); border-radius: 4px; display: inline-block; font-size: 14px;">
                Open in MLflow
            </a>
        </div>
        """.format(run_url), unsafe_allow_html=True)
        
else:
    # Welcome message
    st.markdown("""
    <div class="card" style="text-align: center; padding: 48px 32px;">
        <h2 style="color: var(--text-primary); font-family: 'Inter', sans-serif; margin-bottom: 16px; font-size: 28px; font-weight: 400;">Ready to Start Trading?</h2>
        <p style="color: var(--text-secondary); font-size: 16px; margin-bottom: 32px; line-height: 1.5;">
            Configure your strategy parameters in the sidebar and click <strong>Run Backtest</strong> to begin your algorithmic trading experiment.
        </p>
        <div style="display: flex; justify-content: center; gap: 48px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 32px; margin-bottom: 8px; color: var(--primary-color);">●</div>
                <div style="color: var(--text-secondary); font-size: 14px; font-weight: 500;">Market Data</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 32px; margin-bottom: 8px; color: var(--primary-color);">●</div>
                <div style="color: var(--text-secondary); font-size: 14px; font-weight: 500;">Strategy Config</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 32px; margin-bottom: 8px; color: var(--primary-color);">●</div>
                <div style="color: var(--text-secondary); font-size: 14px; font-weight: 500;">Results Analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
