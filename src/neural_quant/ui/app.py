import streamlit as st
import pandas as pd
import mlflow
from datetime import date
from neural_quant.data.yf_loader import load_yf_data
from neural_quant.strategies.momentum import MovingAverageCrossover
from neural_quant.core.backtest import Backtester

# Page config
st.set_page_config(
    page_title="Neural-Quant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark mode CSS
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    .main .block-container {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e1e1e;
        border-right: 1px solid #333333;
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: #ffffff;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background-color: #262730;
        border: 1px solid #333333;
        color: #ffffff;
        border-radius: 4px;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 1px #1a73e8;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1557b0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main title
st.title("Neural-Quant")
st.markdown("Advanced Algorithmic Trading Platform")

# Sidebar
with st.sidebar:
    st.header("Experiment Configuration")
    
    st.subheader("Market Data")
    ticker = st.text_input("Ticker Symbol", "AAPL")
    start = st.date_input("Start Date", value=date(2021,1,1))
    end = st.date_input("End Date", value=date.today())
    
    st.subheader("Strategy Parameters")
    fast = st.number_input("Fast MA Period", 5, 200, 10, step=1)
    slow = st.number_input("Slow MA Period", 10, 400, 30, step=1)
    threshold = st.number_input("Signal Threshold (%)", 0.0, 5.0, 0.2, step=0.1)
    
    st.subheader("Transaction Costs")
    fee_bps = st.number_input("Commission (bps)", 0, 100, 1, step=1)
    slip_bps = st.number_input("Slippage (bps)", 0, 100, 2, step=1)
    
    st.markdown("---")
    run_btn = st.button("Run Backtest", use_container_width=True)

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ma_crossover")

# Main content
if run_btn:
    with st.spinner("Running backtest..."):
        try:
            # Load data
            df = load_yf_data([ticker], str(start), str(end))
            price = df[f"{ticker}_close"].dropna()

            # Create strategy
            strat = MovingAverageCrossover(ma_fast=fast, ma_slow=slow, threshold=threshold/100.0)
            signals = strat.generate_signals(df)
            
            # Run backtest
            bt = Backtester(commission=fee_bps/10000, slippage=slip_bps/10000)
            results = bt.run_backtest(df, strat)
            equity = results.get('equity_curve', pd.Series())
            trades = results.get('trades', pd.DataFrame())
            metrics = results

            # MLflow logging
            with mlflow.start_run(run_name=f"{ticker}_{fast}-{slow}_{threshold}pct") as run:
                mlflow.log_params({
                    "ticker": ticker, "start": str(start), "end": str(end),
                    "fast": fast, "slow": slow, "threshold_pct": threshold,
                    "fee_bps": fee_bps, "slippage_bps": slip_bps
                })
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))

                # Log artifacts
                equity.to_frame("equity").to_csv("equity.csv")
                trades.to_csv("trades.csv", index=False)
                mlflow.log_artifact("equity.csv")
                mlflow.log_artifact("trades.csv")
                run_url = f"{mlflow.get_tracking_uri().rstrip('/')}/#/experiments/{mlflow.get_experiment_by_name('ma_crossover').experiment_id}/runs/{run.info.run_id}"

            # Display results
            st.success("Backtest completed successfully!")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
            with col4:
                st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
            
            # Charts
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Equity Curve")
                st.line_chart(equity)
            
            with col2:
                st.subheader("All Metrics")
                for key, value in metrics.items():
                    if key not in ['equity_curve', 'trades'] and isinstance(value, (int, float)):
                        st.metric(
                            key.replace('_', ' ').title(),
                            f"{value:.2%}" if 'return' in key or 'drawdown' in key else f"{value:.2f}" if isinstance(value, float) else value
                        )
            
            # Trades table
            st.subheader("Trade Log")
            if not trades.empty:
                st.dataframe(trades, use_container_width=True)
            else:
                st.info("No trades executed during this period.")
            
            # MLflow link
            st.markdown(f"[Open in MLflow]({run_url})")
            
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
else:
    st.info("Configure parameters in the sidebar and click 'Run Backtest' to begin.")