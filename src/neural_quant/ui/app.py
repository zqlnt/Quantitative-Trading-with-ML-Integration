import streamlit as st
import pandas as pd
import mlflow
from datetime import date
from neural_quant.data.yf_loader import load_yf_data
from neural_quant.strategies.momentum import MovingAverageCrossover
from neural_quant.core.backtest import Backtester

st.set_page_config(page_title="Neural-Quant", layout="wide")
st.title("Neural-Quant: Experiment Runner")

with st.sidebar:
    st.header("Experiment")
    ticker = st.text_input("Ticker", "AAPL")
    start = st.date_input("Start", value=date(2021,1,1))
    end   = st.date_input("End",   value=date.today())
    fast = st.number_input("Fast MA", 5, 200, 10, step=1)
    slow = st.number_input("Slow MA", 10, 400, 30, step=1)
    threshold = st.number_input("Signal threshold (%)", 0.0, 5.0, 0.2, step=0.1)
    fee_bps = st.number_input("Fee (bps)", 0, 100, 1, step=1)
    slip_bps = st.number_input("Slippage (bps)", 0, 100, 2, step=1)
    run_btn = st.button("Run Backtest")

# set tracking (adjust if your URI differs)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ma_crossover")

if run_btn:
    with st.spinner("Running..."):
        df = load_yf_data([ticker], str(start), str(end))
        price = df[f"{ticker}_close"].dropna()

        strat = MovingAverageCrossover(ma_fast=fast, ma_slow=slow, threshold=threshold/100.0)
        signals = strat.generate_signals(df)
        bt = Backtester(slippage_bps=slip_bps, fee_bps=fee_bps)
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

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Equity Curve")
            st.line_chart(equity)

        with col2:
            st.subheader("Key Metrics")
            st.json(metrics)

        st.subheader("Trades")
        st.dataframe(trades)

        st.success("Logged to MLflow")
        st.markdown(f"[Open in MLflow]({run_url})")
else:
    st.info("Configure parameters in the sidebar and click **Run Backtest**.")
