# app.py  — Neural Quant Dashboard (MLflow + SQLite)
# Run:  streamlit run app.py
# Env (optional):
#   MLFLOW_DB_PATH=mlflow.db
#   MLFLOW_ARTIFACT_ROOT=./mlruns

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
DB_PATH = os.getenv("MLFLOW_DB_PATH", "mlflow.db")
ARTIFACT_ROOT = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns"))

st.set_page_config(page_title="Neural Quant – Experiments", layout="wide")
st.title("🧠 Neural Quant — Experiment Dashboard")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_runs_from_sqlite(db_path: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    con = sqlite3.connect(db_path)
    try:
        # Pull runs + params + metrics (+ tags if present)
        runs = pd.read_sql_query(
            """
            SELECT r.run_uuid as run_id,
                   r.experiment_id,
                   r.start_time,
                   r.end_time,
                   r.lifecycle_stage
            FROM runs r
            WHERE r.lifecycle_stage='active'
            """,
            con,
        )

        params = pd.read_sql_query(
            "SELECT run_uuid as run_id, key, value FROM params", con
        )
        metrics = pd.read_sql_query(
            "SELECT run_uuid as run_id, key, value FROM metrics", con
        )

        # Optional: tags table may not exist in some backends
        try:
            tags = pd.read_sql_query(
                "SELECT run_uuid as run_id, key, value FROM tags", con
            )
        except Exception:
            tags = pd.DataFrame(columns=["run_id", "key", "value"])

    finally:
        con.close()

    if runs.empty:
        return pd.DataFrame()

    # Pivot params/metrics/tags to columns
    def pivot_kv(df, valcol="value"):
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot_table(index="run_id", columns="key", values=valcol, aggfunc="last")
        wide.columns = [str(c) for c in wide.columns]
        return wide

    P = pivot_kv(params, "value")
    M = pivot_kv(metrics, "value")
    T = pivot_kv(tags, "value")

    df = runs.set_index("run_id").join([P, M, T], how="left")
    df.reset_index(inplace=True)

    # Make numeric where possible
    for c in df.columns:
        if c not in ("run_id", "lifecycle_stage"):
            df[c] = pd.to_numeric(df[c], errors="ignore")

    # Friendly derived columns
    for col in ("start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")

    return df


def artifacts_dir_for_run(run_id: str, experiment_id: str | int) -> Path:
    # MLflow FileStore layout: mlruns/<exp_id>/<run_id>/artifacts
    return ARTIFACT_ROOT / str(experiment_id) / str(run_id) / "artifacts"


@st.cache_data(show_spinner=False)
def load_equity_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path)
            # Attempt to infer ts column
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                df = df.set_index("ts")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# -----------------------------
# Data
# -----------------------------
df = load_runs_from_sqlite(DB_PATH)

if df.empty:
    st.warning(
        f"No MLflow runs found. Ensure MLflow is using SQLite at **{DB_PATH}** "
        "and you have executed at least one experiment."
    )
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
symbols = sorted([s for s in df.get("symbol", pd.Series(dtype=object)).dropna().unique()])
strategies = sorted([s for s in df.get("strategy", pd.Series(dtype=object)).dropna().unique()])
gates = sorted([s for s in df.get("gate", pd.Series(dtype=object)).dropna().unique()])

sym_sel = st.sidebar.multiselect("Symbol", symbols, default=symbols[:1] if symbols else [])
str_sel = st.sidebar.multiselect("Strategy", strategies, default=strategies[:1] if strategies else [])
gate_sel = st.sidebar.multiselect("Gate", gates, default=[])

min_sharpe = st.sidebar.number_input("Min Sharpe (post-cost)", value=-1.0, step=0.1)
max_dd = st.sidebar.number_input("Max Drawdown (≥ negative, e.g., -0.15)", value=0.0, step=0.01, help="Use negative values (e.g., -0.15 for -15%) or 0 to ignore.")

mask = pd.Series(True, index=df.index)
if sym_sel:
    mask &= df["symbol"].isin(sym_sel)
if str_sel:
    mask &= df["strategy"].isin(str_sel)
if gate_sel and "gate" in df.columns:
    mask &= df["gate"].isin(gate_sel)
if "Sharpe_post_cost" in df.columns:
    mask &= df["Sharpe_post_cost"].fillna(-999) >= min_sharpe
if "MaxDD" in df.columns and max_dd < 0:
    mask &= df["MaxDD"].fillna(0) >= max_dd  # remember MaxDD is negative

filtered = df.loc[mask].copy()

# -----------------------------
# Leaderboard + Runs Table
# -----------------------------
st.subheader("Leaderboard")
rank_cols = [c for c in ["run_id", "symbol", "strategy", "ma_fast", "ma_slow",
                         "Sharpe_post_cost", "MaxDD", "Turnover", "VaR95", "CVaR95"] if c in filtered.columns]
if rank_cols:
    lb = filtered[rank_cols].sort_values(by="Sharpe_post_cost", ascending=False, na_position="last")
    st.dataframe(lb, use_container_width=True)
else:
    st.info("No standard metric columns found yet. Run a baseline to populate metrics like Sharpe_post_cost, MaxDD, Turnover.")

st.subheader("All Matching Runs")
st.dataframe(
    filtered.sort_values(by=["start_time"], ascending=False),
    use_container_width=True,
)

# -----------------------------
# Run detail & artifact preview
# -----------------------------
st.subheader("Run Detail & Artifacts")
run_choice = None
if not filtered.empty:
    run_choice = st.selectbox("Select run_id", options=filtered["run_id"].tolist())

if run_choice:
    row = filtered[filtered["run_id"] == run_choice].iloc[0]
    exp_id = str(int(row["experiment_id"]))
    art_dir = artifacts_dir_for_run(run_choice, exp_id)

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Params**")
        param_cols = [c for c in ["symbol","strategy","timeframe","years","ma_fast","ma_slow","fees_bps","slippage_bps"] if c in filtered.columns]
        st.write({k: row.get(k, None) for k in param_cols})

    with cols[1]:
        st.markdown("**Key Metrics**")
        metric_cols = [c for c in ["Sharpe_post_cost","Sortino","MaxDD","Turnover","VaR95","CVaR95","skew","kurtosis","tail_ratio"] if c in filtered.columns]
        st.write({k: row.get(k, None) for k in metric_cols})

    with cols[2]:
        st.markdown("**Meta**")
        st.write({
            "run_id": run_choice,
            "experiment_id": exp_id,
            "start_time": row.get("start_time", ""),
            "end_time": row.get("end_time", ""),
            "gate": row.get("gate", ""),
        })

    st.divider()
    colA, colB = st.columns(2)

    # Try to show equity/price plots if present
    eq_png = next((p for p in ["equity.png","*equity*.png"] if list(art_dir.rglob(p))), None)
    price_png = next((p for p in ["price_ma.png","*price*ma*.png"] if list(art_dir.rglob(p))), None)

    if eq_png:
        # find first match
        eq_path = list(art_dir.rglob(eq_png))[0]
        with colA:
            st.markdown("**Equity Curve**")
            st.image(str(eq_path))

    if price_png:
        price_path = list(art_dir.rglob(price_png))[0]
        with colB:
            st.markdown("**Price & Moving Averages**")
            st.image(str(price_path))

    # Equity CSV preview + chart
    eq_csv = next((p for p in ["equity.csv","*equity*.csv"] if list(art_dir.rglob(p))), None)
    if eq_csv:
        eq_path = list(art_dir.rglob(eq_csv))[0]
        df_eq = load_equity_csv(eq_path)
        if not df_eq.empty:
            st.markdown("**Equity Series (CSV)**")
            st.dataframe(df_eq.tail(20), use_container_width=True)
            # Show a quick line chart if we can infer a column
            ycol = "equity" if "equity" in df_eq.columns else df_eq.columns[-1]
            st.line_chart(df_eq[ycol])

    # List all artifacts
    if art_dir.exists():
        with st.expander("Artifact files"):
            files = [str(p.relative_to(art_dir)) for p in art_dir.rglob("*") if p.is_file()]
            if files:
                st.code("\n".join(files))
            else:
                st.write("No files in artifacts directory.")

st.caption(f"DB: {DB_PATH} • Artifacts: {ARTIFACT_ROOT}")
