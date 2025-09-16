import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from neural_quant.data.yf_loader import load_yf_data
from neural_quant.strategies.strategy_registry import get_strategy_registry, get_strategy
from neural_quant.core.backtest import Backtester
from neural_quant.core.portfolio_backtest import PortfolioBacktester
from neural_quant.utils.llm_assistant import NeuralQuantAssistant
from neural_quant.analysis.walkforward import WalkForwardAnalyzer, WalkForwardConfig
from neural_quant.analysis.allocation_methods import AllocationMethodConfig
from neural_quant.analysis.position_management import PositionManagementConfig
from neural_quant.analysis.basic_exits import BasicExitsConfig

# Page config
st.set_page_config(
    page_title="Neural-Quant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styled components
st.markdown("""
<style>
/* Dark mode theme */
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

/* Overall UI scaling and zoom */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 95%;
    background-color: #0e1117;
}

/* Sidebar scaling */
.sidebar .sidebar-content {
    padding-top: 1rem;
    background-color: #1e1e1e;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

.metric-card.good {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
}

.metric-card.bad {
    background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
}

.metric-card.neutral {
    background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.85rem;
    opacity: 0.9;
}

.trade-pnl-positive {
    color: #4CAF50;
    font-weight: bold;
}

.trade-pnl-negative {
    color: #f44336;
    font-weight: bold;
}

.quick-prompt-btn {
    margin: 0.2rem;
    width: 100%;
}

/* Market Watch ticker containers - Dark Mode */
.stContainer {
    border: 2px solid #404040;
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    background: #1e1e1e;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    transition: all 0.2s ease;
}

.stContainer:hover {
    border-color: #667eea;
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    transform: translateY(-1px);
    background: #2a2a2a;
}

/* Market Watch ticker boxes - Dark Mode */
.ticker-box {
    border: 2px solid #404040;
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    background: #1e1e1e;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    transition: all 0.2s ease;
}

.ticker-box:hover {
    border-color: #667eea;
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    transform: translateY(-1px);
    background: #2a2a2a;
}

.ticker-header {
    font-weight: bold;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
    color: #fafafa;
    text-align: center;
}

.ticker-chart {
    margin: 0.5rem 0;
}

.ticker-metric {
    font-size: 0.8rem;
    color: #cccccc;
    text-align: center;
    margin: 0.25rem 0;
}

/* Dark mode for Streamlit components */
.stSelectbox > div > div {
    background-color: #1e1e1e;
    color: #fafafa;
}

.stTextInput > div > div > input {
    background-color: #1e1e1e;
    color: #fafafa;
    border-color: #404040;
}

.stNumberInput > div > div > input {
    background-color: #1e1e1e;
    color: #fafafa;
    border-color: #404040;
}

.stDateInput > div > div > input {
    background-color: #1e1e1e;
    color: #fafafa;
    border-color: #404040;
}

.stButton > button {
    background-color: #667eea;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #5a6fd8;
}

/* Responsive grid adjustments */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background-color: #1e1e1e;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    background-color: #1e1e1e;
    color: #fafafa;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #667eea;
    color: white;
}

/* Chart height adjustments */
.stLineChart {
    height: 120px !important;
}

/* Overall font scaling */
.stApp {
    font-size: 0.9rem;
    color: #fafafa;
}

/* Sidebar font scaling */
.sidebar .sidebar-content {
    font-size: 0.9rem;
    color: #fafafa;
}

/* Main content font scaling */
.main .block-container {
    font-size: 0.9rem;
    color: #fafafa;
}

/* Dark mode for dataframes */
.stDataFrame {
    background-color: #1e1e1e;
    color: #fafafa;
}

/* Dark mode for expanders */
.stExpander {
    background-color: #1e1e1e;
    border-color: #404040;
}

.stExpander > div > div {
    background-color: #1e1e1e;
    color: #fafafa;
}

/* Dark mode for metrics */
.stMetric {
    background-color: #1e1e1e;
    color: #fafafa;
}

/* Dark mode for info boxes */
.stInfo {
    background-color: #1e1e1e;
    border-color: #404040;
    color: #fafafa;
}

.stWarning {
    background-color: #1e1e1e;
    border-color: #ffa726;
    color: #fafafa;
}

.stError {
    background-color: #1e1e1e;
    border-color: #f44336;
    color: #fafafa;
}

.stSuccess {
    background-color: #1e1e1e;
    border-color: #4caf50;
    color: #fafafa;
}
</style>
""", unsafe_allow_html=True)

def create_metric_card(title, value, format_type="percentage", threshold_good=0, threshold_bad=-0.1):
    """Create a styled metric card with color coding."""
    if format_type == "percentage":
        display_value = f"{value:.2%}"
    elif format_type == "number":
        display_value = f"{value:.2f}"
    else:
        display_value = str(value)
    
    # Determine card class based on value and thresholds
    if format_type == "percentage" and value >= threshold_good:
        card_class = "good"
    elif format_type == "percentage" and value <= threshold_bad:
        card_class = "bad"
    else:
        card_class = "neutral"
    
    return f"""
    <div class="metric-card {card_class}">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{display_value}</div>
    </div>
    """

def create_trade_pnl_chart(trades_df):
    """Create a PnL bar chart for trades."""
    if trades_df.empty or 'pnl' not in trades_df.columns:
        return None
    
    # Create bar chart with color coding
    colors = ['#4CAF50' if pnl > 0 else '#f44336' for pnl in trades_df['pnl']]
    
    fig = go.Figure(data=go.Bar(
        x=list(range(len(trades_df))),
        y=trades_df['pnl'],
        marker_color=colors,
        text=[f"Trade {i+1}" for i in range(len(trades_df))],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Trade P&L",
        xaxis_title="Trade Number",
        yaxis_title="P&L ($)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_trade_heatmap(trades_df):
    """Create a heatmap showing trade performance by day of week and hour."""
    if trades_df.empty or 'entry_time' not in trades_df.columns:
        return None
    
    # Convert to datetime if needed
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['day_of_week'] = trades_df['entry_time'].dt.day_name()
    trades_df['hour'] = trades_df['entry_time'].dt.hour
    
    # Create pivot table for heatmap
    heatmap_data = trades_df.groupby(['day_of_week', 'hour'])['pnl'].mean().unstack(fill_value=0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order, fill_value=0)
    
    fig = px.imshow(
        heatmap_data.values,
        labels=dict(x="Hour", y="Day of Week", color="Avg P&L"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    
    fig.update_layout(
        title="Trade Performance Heatmap (P&L by Day/Hour)",
        height=400
    )
    
    return fig

# Main title
st.title("Neural-Quant")
st.markdown("Advanced Algorithmic Trading Platform with AI Assistant")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'assistant' not in st.session_state:
    try:
        st.session_state.assistant = NeuralQuantAssistant()
        st.session_state.assistant_available = True
    except ValueError:
        st.session_state.assistant = None
        st.session_state.assistant_available = False

# Get strategy registry
strategy_registry = get_strategy_registry()

# Sidebar
with st.sidebar:
    st.header("Experiment Configuration")
    
    st.subheader("Strategy Selection")
    strategy_name = st.selectbox(
        "Select Strategy",
        options=strategy_registry.list_strategies(),
        index=0,
        help="Choose a trading strategy to backtest"
    )
    
    # Get strategy info
    strategy_info = strategy_registry.get_strategy_info(strategy_name)
    is_portfolio_strategy = strategy_info.get('is_portfolio_strategy', False)
    
    st.subheader("Market Data")
    
    if is_portfolio_strategy:
        # Portfolio strategies - multiple tickers
        st.write("**Portfolio Strategy - Select Multiple Tickers**")
        available_tickers = [
            "AAPL", "TSLA", "NVDA", "PLTR", "GOOG", "LMT", "NOC", 
            "BTC-USD", "SOL-USD", "XAUUSD=X", "CL=F", "GBPUSD=X"
        ]
        
        selected_tickers = st.multiselect(
            "Select Tickers",
            options=available_tickers,
            default=["AAPL", "TSLA", "NVDA"],
            help="Select multiple tickers for portfolio strategy"
        )
        
        if not selected_tickers:
            st.error("Please select at least one ticker for portfolio strategy")
            selected_tickers = ["AAPL"]  # Default fallback
    else:
        # Single ticker strategies
        ticker = st.text_input("Ticker Symbol", "AAPL")
        selected_tickers = [ticker]
    
    start = st.date_input("Start Date", value=date(2021,1,1))
    end = st.date_input("End Date", value=date.today())
    
    st.subheader("Strategy Parameters")
    
    # Dynamic parameter inputs based on selected strategy
    strategy_params = {}
    
    if strategy_name == "momentum" or strategy_name == "moving_average_crossover":
        strategy_params['ma_fast'] = st.number_input("Fast MA Period", 5, 200, 10, step=1)
        strategy_params['ma_slow'] = st.number_input("Slow MA Period", 10, 400, 30, step=1)
        strategy_params['threshold'] = st.number_input("Signal Threshold (%)", 0.0, 5.0, 0.2, step=0.1) / 100.0
        strategy_params['min_volume'] = st.number_input("Min Volume", 0, 10000000, 1000000, step=100000)
        strategy_params['max_positions'] = st.number_input("Max Positions", 1, 20, 5, step=1)
    
    elif strategy_name == "bollinger_bands":
        strategy_params['window'] = st.number_input("Window", 5, 100, 20, step=1)
        strategy_params['num_std'] = st.number_input("Standard Deviations", 0.5, 5.0, 2.0, step=0.1)
    
    elif strategy_name == "volatility_breakout":
        strategy_params['atr_window'] = st.number_input("ATR Window", 5, 50, 14, step=1)
        strategy_params['multiplier'] = st.number_input("ATR Multiplier", 0.5, 5.0, 1.5, step=0.1)
    
    elif strategy_name == "cross_sectional_momentum":
        strategy_params['lookback_window'] = st.number_input("Lookback Window", 5, 100, 20, step=1)
        strategy_params['top_n'] = st.number_input("Top N (Long)", 1, 10, 3, step=1)
        strategy_params['bottom_n'] = st.number_input("Bottom N (Short)", 1, 10, 3, step=1)
    
    st.subheader("Transaction Costs")
    fee_bps = st.number_input("Commission (bps)", 0, 100, 1, step=1)
    slip_bps = st.number_input("Slippage (bps)", 0, 100, 2, step=1)
    
    if is_portfolio_strategy:
        max_positions = st.number_input("Max Portfolio Positions", 1, 20, 12, step=1)
    
    st.subheader("Statistical Analysis")
    enable_mcpt = st.checkbox("Enable Monte Carlo Permutation Testing", value=True, 
                             help="Test statistical significance of backtest results")
    
    if enable_mcpt:
        mcpt_permutations = st.number_input("Number of Permutations", 100, 5000, 1000, step=100,
                                           help="More permutations = more accurate but slower")
        mcpt_block_size = st.number_input("Block Size (for autocorrelation)", 1, 50, 5, step=1,
                                         help="Preserve autocorrelation in returns. Set to 1 for simple shuffle.")
        mcpt_significance = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01,
                                     help="P-value threshold for significance (0.05 = 5%)")
    
    enable_bootstrap = st.checkbox("Enable Bootstrap Confidence Intervals", value=True,
                                  help="Compute confidence intervals by resampling trade P&L")
    
    if enable_bootstrap:
        bootstrap_samples = st.number_input("Number of Bootstrap Samples", 100, 5000, 1000, step=100,
                                           help="More samples = more accurate confidence intervals")
        bootstrap_confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01,
                                        help="Confidence level for intervals (0.95 = 95%)")
        bootstrap_method = st.selectbox("Resampling Method", 
                                       ["trades", "returns"],
                                       help="Resample trades (P&L) or returns series")
    
    st.markdown("---")
    st.subheader("Regime Filter")
    
    # Regime filter toggle
    enable_regime_filter = st.checkbox("Enable Regime Filter", 
                                     help="Only trade when market proxy is in specified regime")
    
    if enable_regime_filter:
        col1, col2 = st.columns(2)
        
        with col1:
            regime_proxy = st.text_input("Market Proxy Symbol", 
                                       value="SPY", 
                                       help="Symbol to use as market proxy (e.g., SPY, ^GSPC)")
        
        with col2:
            regime_rule = st.selectbox("Regime Rule",
                                     ["Bull only (proxy > SMA(200))", 
                                      "Bear only (proxy < SMA(200))", 
                                      "Both (no filter)"],
                                     help="Trading rule based on proxy regime")
    
    st.markdown("---")
    st.subheader("Volatility Targeting")
    
    # Volatility targeting toggle
    enable_vol_targeting = st.checkbox("Enable Volatility Targeting", 
                                     help="Scale portfolio exposure to hit target annual volatility")
    
    if enable_vol_targeting:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_vol = st.number_input("Target Volatility (%)", 
                                       min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                                       help="Target annual volatility percentage") / 100.0
        
        with col2:
            lookback_window = st.number_input("Lookback Window (days)", 
                                            min_value=5, max_value=100, value=20, step=1,
                                            help="Window for realized volatility calculation")
        
        with col3:
            scale_cap = st.number_input("Scale Cap", 
                                      min_value=0.5, max_value=5.0, value=2.0, step=0.1,
                                      help="Maximum scaling factor")
    
    st.markdown("---")
    st.subheader("Portfolio Allocation")
    
    # Allocation method control
    allocation_method = st.selectbox("Allocation Method",
                                   ["Equal Weight", "Volatility Weighted"],
                                   help="Method for allocating portfolio weights")
    
    if allocation_method == "Volatility Weighted":
        vol_lookback = st.number_input("Volatility Lookback (days)", 
                                     min_value=5, max_value=100, value=20, step=1,
                                     help="Window for volatility calculation")
    
    st.markdown("---")
    st.subheader("Position Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position_pct = st.number_input("Max Position per Name (%)", 
                                         min_value=1.0, max_value=50.0, value=15.0, step=1.0,
                                         help="Maximum position size per individual asset") / 100.0
    
    with col2:
        rebalance_frequency = st.selectbox("Rebalance Frequency",
                                         ["Daily", "Weekly", "Monthly"],
                                         index=2,  # Default to Monthly
                                         help="How often to rebalance the portfolio")
    
    st.markdown("---")
    st.subheader("Basic Exits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_atr_stop = st.checkbox("Enable ATR Stop", 
                                    help="Exit positions when price crosses trailing ATR stop")
        if enable_atr_stop:
            atr_window = st.number_input("ATR Window", 
                                       min_value=5, max_value=50, value=14, step=1,
                                       help="Window for ATR calculation")
            atr_multiplier = st.number_input("ATR Multiplier", 
                                           min_value=1.0, max_value=5.0, value=2.5, step=0.1,
                                           help="ATR stop distance multiplier")
    
    with col2:
        enable_time_stop = st.checkbox("Enable Time Stop", 
                                     help="Exit positions after specified number of bars")
        if enable_time_stop:
            time_stop_bars = st.number_input("Time Stop (bars)", 
                                           min_value=5, max_value=100, value=30, step=1,
                                           help="Number of bars to hold position")
    
    st.markdown("---")
    run_btn = st.button("Run Backtest", width='stretch')
    
    # AI Assistant Section
    st.markdown("---")
    st.header("AI Assistant")
    
    if not st.session_state.assistant_available:
        st.warning("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable to enable AI features.")
    else:
        st.success("AI Assistant Ready")
        
        # Quick analysis button
        if st.session_state.current_results:
            if st.button("Analyze Results", width='stretch'):
                with st.spinner("AI is analyzing your results..."):
                    analysis = st.session_state.assistant.analyze_backtest_results(
                        st.session_state.current_results['metrics'],
                        st.session_state.current_results['equity'],
                        st.session_state.current_results['trades'],
                        st.session_state.current_results['params']
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"## Backtest Analysis\n\n{analysis}"
                    })
                    st.rerun()
        
        # Quick prompt buttons
        st.subheader("Quick Prompts")
        quick_prompts = [
            "Summarize performance",
            "Suggest parameter improvements", 
            "What if I traded only during high volatility?",
            "Compare to buy & hold",
            "Identify best/worst trades",
            "Risk analysis and recommendations"
        ]
        
        for i, prompt in enumerate(quick_prompts):
            if st.button(prompt, key=f"quick_prompt_{i}", width='stretch'):
                with st.spinner("AI is thinking..."):
                    context = st.session_state.current_results if st.session_state.current_results else None
                    answer = st.session_state.assistant.answer_question(prompt, context)
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": prompt
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer
                    })
                    st.rerun()
        
        # Chat interface
        st.subheader("Ask Questions")
        user_question = st.text_input("Ask about trading, strategies, or results:", key="user_question")
        
        if st.button("Send", key="send_question") and user_question:
            with st.spinner("AI is thinking..."):
                context = st.session_state.current_results if st.session_state.current_results else None
                answer = st.session_state.assistant.answer_question(user_question, context)
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer
                })
                st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat", width='stretch'):
            st.session_state.chat_history = []
            st.rerun()

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ma_crossover")

# Main content area with tabs
st.markdown("<br>", unsafe_allow_html=True)  # Add some top spacing
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Backtest Results", "Walk-Forward Analysis", "Strategy Library", "Metrics Key", "Market Watch"])

with tab1:
    if run_btn:
        with st.spinner("Running backtest..."):
            try:
                # Load data for all selected tickers
                data_dict = {}
                for ticker in selected_tickers:
                    df = load_yf_data([ticker], str(start), str(end))
                    data_dict[ticker] = df
                
                # Create strategy
                strat = get_strategy(strategy_name, **strategy_params)
                
                # Create MCPT configuration if enabled
                mcpt_config = None
                if enable_mcpt:
                    from neural_quant.analysis.mcpt import MCPTConfig
                    mcpt_config = MCPTConfig(
                        n_permutations=mcpt_permutations,
                        block_size=mcpt_block_size if mcpt_block_size > 1 else None,
                        significance_level=mcpt_significance
                    )
                
                # Create Bootstrap configuration if enabled
                bootstrap_config = None
                if enable_bootstrap:
                    from neural_quant.analysis.bootstrap import BootstrapConfig
                    bootstrap_config = BootstrapConfig(
                        n_bootstrap=bootstrap_samples,
                        confidence_level=bootstrap_confidence,
                        resample_method=bootstrap_method
                    )
                
                # Create regime filter configuration if enabled
                regime_filter_config = None
                if enable_regime_filter:
                    from neural_quant.analysis.regime_filter import create_regime_filter_config
                    regime_filter_config = create_regime_filter_config(
                        enabled=True,
                        proxy_symbol=regime_proxy,
                        regime_rule=regime_rule
                    )
                
                # Create volatility targeting configuration if enabled
                vol_targeting_config = None
                if enable_vol_targeting:
                    from neural_quant.analysis.volatility_targeting import create_volatility_targeting_config
                    vol_targeting_config = create_volatility_targeting_config(
                        enabled=True,
                        target_vol=target_vol,
                        lookback_window=lookback_window,
                        scale_cap=scale_cap
                    )
                
                # Run appropriate backtest based on strategy type
                if is_portfolio_strategy:
                    # Portfolio backtest
                    # Create allocation method config
                    allocation_method_lower = allocation_method.lower().replace(" ", "_")
                    allocation_config = AllocationMethodConfig(
                        method=allocation_method_lower,
                        vol_lookback=vol_lookback if allocation_method == "Volatility Weighted" else 20
                    )
                    
                    # Create position management config
                    rebalance_freq_lower = rebalance_frequency.lower()
                    position_config = PositionManagementConfig(
                        max_position_pct=max_position_pct,
                        rebalance_frequency=rebalance_freq_lower,
                        min_rebalance_interval=1,
                        turnover_threshold=0.05
                    )
                    
                    # Create basic exits config
                    basic_exits_config = BasicExitsConfig(
                        enable_atr_stop=enable_atr_stop,
                        atr_window=atr_window if enable_atr_stop else 14,
                        atr_multiplier=atr_multiplier if enable_atr_stop else 2.5,
                        enable_time_stop=enable_time_stop,
                        time_stop_bars=time_stop_bars if enable_time_stop else 30
                    )
                    
                    bt = PortfolioBacktester(
                        commission=fee_bps/10000, 
                        slippage=slip_bps/10000,
                        max_positions=max_positions,
                        enable_mcpt=enable_mcpt,
                        mcpt_config=mcpt_config,
                        enable_bootstrap=enable_bootstrap,
                        bootstrap_config=bootstrap_config,
                        enable_regime_filter=enable_regime_filter,
                        regime_filter_config=regime_filter_config,
                        enable_vol_targeting=enable_vol_targeting,
                        vol_targeting_config=vol_targeting_config,
                        allocation_method=allocation_method_lower,
                        allocation_config=allocation_config,
                        position_management_config=position_config,
                        enable_basic_exits=enable_atr_stop or enable_time_stop,
                        basic_exits_config=basic_exits_config
                    )
                    results = bt.run_portfolio_backtest(data_dict, strat, str(start), str(end))
                else:
                    # Single ticker backtest
                    if len(selected_tickers) > 1:
                        st.warning("Single-ticker strategy selected but multiple tickers chosen. Using first ticker only.")
                    
                    ticker = selected_tickers[0]
                    df = data_dict[ticker]
                    
                    # Prepare data for single-ticker strategy by renaming columns
                    df_prepared = df.copy()
                    if f"{ticker}_close" in df_prepared.columns:
                        df_prepared = df_prepared.rename(columns={
                            f"{ticker}_close": "close",
                            f"{ticker}_open": "open",
                            f"{ticker}_high": "high",
                            f"{ticker}_low": "low",
                            f"{ticker}_volume": "volume"
                        })
                    
                    # Create regime filter configuration if enabled
                    regime_filter_config = None
                    if enable_regime_filter:
                        from neural_quant.analysis.regime_filter import create_regime_filter_config
                        regime_filter_config = create_regime_filter_config(
                            enabled=True,
                            proxy_symbol=regime_proxy,
                            regime_rule=regime_rule
                        )
                    
                    # Create volatility targeting configuration if enabled
                    vol_targeting_config = None
                    if enable_vol_targeting:
                        from neural_quant.analysis.volatility_targeting import create_volatility_targeting_config
                        vol_targeting_config = create_volatility_targeting_config(
                            enabled=True,
                            target_vol=target_vol,
                            lookback_window=lookback_window,
                            scale_cap=scale_cap
                        )
                    
                    bt = Backtester(
                        commission=fee_bps/10000, 
                        slippage=slip_bps/10000,
                        enable_mcpt=enable_mcpt,
                        mcpt_config=mcpt_config,
                        enable_bootstrap=enable_bootstrap,
                        bootstrap_config=bootstrap_config,
                        enable_regime_filter=enable_regime_filter,
                        regime_filter_config=regime_filter_config,
                        enable_vol_targeting=enable_vol_targeting,
                        vol_targeting_config=vol_targeting_config
                    )
                    
                    results = bt.run_backtest(df_prepared, strat, str(start), str(end))
                
                # Extract results
                equity = results.get('equity_curve', pd.Series())
                trades = results.get('trades', [])
                metrics = results

                # MLflow logging is handled by the backtester
                # No need for duplicate logging here

                # Store results for AI assistant
                st.session_state.current_results = {
                    'metrics': metrics,
                    'equity': equity,
                    'trades': trades,
                    'params': {
                        'strategy': strategy_name,
                        'tickers': selected_tickers,
                        'start': str(start),
                        'end': str(end),
                        'fee_bps': fee_bps,
                        'slippage_bps': slip_bps,
                        **strategy_params
                    }
                }
                
                # Display results
                st.success("Backtest completed successfully!")
                
                # Display regime filter information if enabled
                if enable_regime_filter and 'regime_hit_rate' in metrics:
                    st.info(f"🎯 **Regime Filter Active**: {regime_rule} using {regime_proxy} | "
                           f"Trading Days: {metrics.get('regime_hit_rate', 0):.1%} "
                           f"({metrics.get('regime_trading_days', 0)}/{metrics.get('regime_total_days', 0)})")
                
                # Display volatility targeting information if enabled
                if enable_vol_targeting and 'volatility_targeting' in results:
                    vol_info = results['volatility_targeting']
                    st.info(f"📊 **Volatility Targeting Active**: Target {vol_info['target_vol']:.1%} | "
                           f"Pre: {vol_info['realized_vol_pre']:.1%} → Post: {vol_info['realized_vol_post']:.1%} | "
                           f"Avg Scaling: {vol_info['avg_scaling']:.2f}x")
                
                # Styled metric cards
                st.subheader("Performance Dashboard")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(create_metric_card(
                        "Total Return", 
                        metrics.get('total_return', 0), 
                        "percentage", 
                        threshold_good=0.1, 
                        threshold_bad=-0.1
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(create_metric_card(
                        "Sharpe Ratio", 
                        metrics.get('sharpe_ratio', 0), 
                        "number", 
                        threshold_good=1.0, 
                        threshold_bad=0.0
                    ), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(create_metric_card(
                        "Max Drawdown", 
                        metrics.get('max_drawdown', 0), 
                        "percentage", 
                        threshold_good=-0.05, 
                        threshold_bad=-0.2
                    ), unsafe_allow_html=True)
                
                with col4:
                    st.markdown(create_metric_card(
                        "Total Trades", 
                        metrics.get('total_trades', 0), 
                        "integer"
                    ), unsafe_allow_html=True)
                
                # Additional metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(create_metric_card(
                        "Win Rate", 
                        metrics.get('win_rate', 0), 
                        "percentage", 
                        threshold_good=0.5, 
                        threshold_bad=0.3
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(create_metric_card(
                        "Profit Factor", 
                        metrics.get('profit_factor', 0), 
                        "number", 
                        threshold_good=1.5, 
                        threshold_bad=1.0
                    ), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(create_metric_card(
                        "Volatility", 
                        metrics.get('volatility', 0), 
                        "percentage", 
                        threshold_good=0.15, 
                        threshold_bad=0.3
                    ), unsafe_allow_html=True)
                
                with col4:
                    st.markdown(create_metric_card(
                        "CAGR", 
                        metrics.get('cagr', 0), 
                        "percentage", 
                        threshold_good=0.1, 
                        threshold_bad=-0.1
                    ), unsafe_allow_html=True)
                
                # Ask About This Run Button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🤖 Ask About This Run", help="Ask questions about this backtest run with AI assistance", use_container_width=True):
                        st.session_state['show_qa'] = True
                        st.session_state['qa_results'] = results
                        st.session_state['qa_strategy'] = strategy_name
                        st.session_state['qa_tickers'] = selected_tickers
                
                # Q&A Section
                if st.session_state.get('show_qa', False) and st.session_state.get('qa_results') == results:
                    st.markdown("### 🤖 AI Assistant - Ask About This Run")
                    
                    # Load Q&A system
                    from neural_quant.analysis.run_qa import RunQASystem
                    qa_system = RunQASystem()
                    
                    # Create context
                    context = qa_system.load_run_context(
                        run_id="current_run",
                        artifacts={
                            'params': {
                                'strategy': st.session_state.get('qa_strategy', 'Unknown'),
                                'tickers': st.session_state.get('qa_tickers', []),
                                'start_date': str(start),
                                'end_date': str(end)
                            },
                            'metrics': results.get('metrics', {}),
                            'trades': results.get('trades', []),
                            'mcpt_results': results.get('mcpt_results', {}),
                            'bootstrap_results': results.get('bootstrap_results', {})
                        }
                    )
                    
                    # Common questions
                    st.markdown("**Common Questions:**")
                    common_questions = qa_system.get_common_questions()
                    cols = st.columns(3)
                    for i, question in enumerate(common_questions[:6]):  # Show first 6 questions
                        with cols[i % 3]:
                            if st.button(question, key=f"common_q_{i}"):
                                st.session_state['qa_question'] = question
                    
                    # Custom question input
                    question = st.text_input("Ask a custom question:", value=st.session_state.get('qa_question', ''))
                    
                    if st.button("Ask Question") and question:
                        with st.spinner("Analyzing your question..."):
                            answer = qa_system.answer_question(question, context)
                            st.markdown(f"**Answer:** {answer}")
                    
                    if st.button("Close Q&A"):
                        st.session_state['show_qa'] = False
                        st.rerun()
                
                # Quant Researcher Button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🧠 Generate Experiments", help="Generate testable experiments based on this run's performance", use_container_width=True):
                        st.session_state['show_researcher'] = True
                        st.session_state['researcher_results'] = results
                        st.session_state['researcher_strategy'] = strategy_name
                        st.session_state['researcher_tickers'] = selected_tickers
                
                # Quant Researcher Section
                if st.session_state.get('show_researcher', False) and st.session_state.get('researcher_results') == results:
                    st.markdown("### 🧠 Quant Researcher - Generate Experiments")
                    
                    # Load Quant Researcher
                    from neural_quant.analysis.quant_researcher import QuantResearcher
                    researcher = QuantResearcher()
                    
                    # Create artifacts for analysis
                    artifacts = {
                        'metrics': results.get('metrics', {}),
                        'params': {
                            'strategy': st.session_state.get('researcher_strategy', 'Unknown'),
                            'tickers': st.session_state.get('researcher_tickers', []),
                            'start_date': str(start),
                            'end_date': str(end)
                        },
                        'mcpt_results': results.get('mcpt_results', {}),
                        'bootstrap_results': results.get('bootstrap_results', {}),
                        'walkforward_results': results.get('walkforward_results', {})
                    }
                    
                    # Generate experiments
                    with st.spinner("Analyzing run artifacts and generating experiments..."):
                        experiments = researcher.generate_experiments(
                            artifacts=artifacts,
                            universe=st.session_state.get('researcher_tickers', []),
                            max_experiments=3
                        )
                    
                    # Display hypotheses
                    if experiments.get('hypotheses'):
                        st.subheader("Research Hypotheses")
                        for hypothesis in experiments['hypotheses']:
                            with st.expander(f"**{hypothesis['id']}**: {hypothesis['rationale'][:100]}..."):
                                st.write(f"**Rationale:** {hypothesis['rationale']}")
                                st.write(f"**Regimes:** {', '.join(hypothesis['regimes'])}")
                                st.write(f"**Tickers:** {', '.join(hypothesis['tickers'])}")
                    
                    # Display experiments
                    if experiments.get('experiments'):
                        st.subheader("Proposed Experiments")
                        for i, experiment in enumerate(experiments['experiments']):
                            with st.expander(f"**{experiment['id']}**: {experiment['strategy']} Strategy"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Strategy Parameters:**")
                                    for param, value in experiment['params'].items():
                                        st.write(f"- {param}: {value}")
                                
                                with col2:
                                    st.write("**Overlays:**")
                                    overlays = experiment['overlays']
                                    st.write(f"- Regime: {overlays['regime']}")
                                    st.write(f"- Vol Target: {overlays['vol_target']}")
                                    st.write(f"- Allocation: {overlays['alloc']}")
                                    st.write(f"- Position Cap: {overlays['position_cap']}")
                                
                                st.write("**Success Criteria:**")
                                criteria = experiment['success_criteria']
                                for criterion, threshold in criteria.items():
                                    st.write(f"- {criterion}: {threshold}")
                                
                                st.write("**Risks:**")
                                for risk in experiment['risks']:
                                    st.write(f"- {risk}")
                    
                    # Display data needs
                    if experiments.get('next_data_needs'):
                        st.subheader("Next Data Needs")
                        for need in experiments['next_data_needs']:
                            st.write(f"- {need}")
                    
                    # Show raw JSON
                    with st.expander("Raw JSON Output"):
                        json_output = researcher.format_as_json(experiments)
                        st.code(json_output, language='json')
                    
                    # Strategy Developer Button
                    if experiments.get('experiments'):
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("🔧 Generate Changes", help="Translate experiments into precise change requests", use_container_width=True):
                                st.session_state['show_developer'] = True
                                st.session_state['developer_experiments'] = experiments['experiments']
                    
                    # Strategy Developer Section
                    if st.session_state.get('show_developer', False) and st.session_state.get('developer_experiments'):
                        st.markdown("### 🔧 Strategy Developer - Generate Changes")
                        
                        # Load Strategy Developer
                        from neural_quant.analysis.strategy_developer import StrategyDeveloper
                        developer = StrategyDeveloper()
                        
                        # Let user select experiments
                        st.subheader("Select Experiments to Implement")
                        selected_experiments = []
                        
                        for i, experiment in enumerate(st.session_state.get('developer_experiments', [])):
                            if st.checkbox(f"**{experiment['id']}**: {experiment['strategy']} - {experiment.get('rationale', 'No rationale')[:100]}...", 
                                         key=f"exp_select_{i}"):
                                selected_experiments.append(experiment)
                        
                        if selected_experiments and st.button("Generate Change Requests"):
                            with st.spinner("Translating experiments into change requests..."):
                                changes = developer.translate_experiments(
                                    selected_experiments=selected_experiments,
                                    max_grid_size=9
                                )
                            
                            # Display changes
                            if changes.get('changes'):
                                st.subheader("Proposed Changes")
                                for i, change in enumerate(changes['changes']):
                                    with st.expander(f"**Change {i+1}**: {change['type'].replace('_', ' ').title()}"):
                                        if change['type'] == 'param_grid':
                                            st.write(f"**Strategy:** {change['strategy']}")
                                            st.write("**Parameter Grid:**")
                                            for param, values in change['grid'].items():
                                                st.write(f"- {param}: {values}")
                                        
                                        elif change['type'] in ['overlay_update', 'overlay_add']:
                                            st.write(f"**Overlay:** {change['name']}")
                                            st.write("**Settings:**")
                                            for setting, value in change['settings'].items():
                                                st.write(f"- {setting}: {value}")
                                        
                                        elif change['type'] == 'code_change':
                                            st.write(f"**File:** {change['file']}")
                                            st.write(f"**Function:** {change['function']}")
                                            st.write(f"**Description:** {change['description']}")
                            
                            # Display sweeps
                            if changes.get('sweeps'):
                                st.subheader("Parameter Sweeps")
                                for i, sweep in enumerate(changes['sweeps']):
                                    with st.expander(f"**Sweep {i+1}**: {sweep['strategy']} (Max {sweep['max_grid']} combinations)"):
                                        st.write("**Parameters:**")
                                        for param, values in sweep['params'].items():
                                            st.write(f"- {param}: {values}")
                            
                            # Display tests
                            if changes.get('tests'):
                                st.subheader("Recommended Tests")
                                for test in changes['tests']:
                                    st.write(f"- {test}")
                            
                            # Show raw JSON
                            with st.expander("Raw JSON Output"):
                                json_output = developer.format_as_json(changes)
                                st.code(json_output, language='json')
                        
                        if st.button("Close Developer"):
                            st.session_state['show_developer'] = False
                            st.rerun()
                    
                    if st.button("Close Researcher"):
                        st.session_state['show_researcher'] = False
                        st.rerun()
                
                # MCPT Significance Testing Results
                if 'mcpt_results' in results and 'summary' in results['mcpt_results']:
                    st.subheader("Statistical Significance Analysis")
                    st.markdown("**Monte Carlo Permutation Test Results** - Testing if observed performance could have occurred by chance")
                    
                    mcpt_summary = results['mcpt_results']['summary']
                    if not mcpt_summary.empty:
                        # Display MCPT results in a clean table
                        st.dataframe(
                            mcpt_summary.style.format({
                                'Observed': '{:.4f}',
                                'Null Mean': '{:.4f}',
                                'Null Std': '{:.4f}',
                                'P-Value': '{:.4f}',
                                'CI Lower': '{:.4f}',
                                'CI Upper': '{:.4f}'
                            }).map(
                                lambda x: 'background-color: #4CAF50; color: white' if x == True else 
                                        'background-color: #f44336; color: white' if x == False else '',
                                subset=['Significant']
                            ),
                            width='stretch'
                        )
                        
                        # Significance summary
                        significant_metrics = mcpt_summary[mcpt_summary['Significant'] == True]
                        total_metrics = len(mcpt_summary)
                        significant_count = len(significant_metrics)
                        
                        if significant_count > 0:
                            st.success(f"✅ {significant_count}/{total_metrics} metrics show statistical significance (p < 0.05)")
                            if significant_count == total_metrics:
                                st.balloons()
                        else:
                            st.warning(f"⚠️ {significant_count}/{total_metrics} metrics show statistical significance. Results may be due to chance.")
                        
                        # Key insights
                        st.subheader("Key Insights")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Most Significant Metrics:**")
                            top_significant = significant_metrics.nsmallest(3, 'P-Value')
                            for _, row in top_significant.iterrows():
                                st.write(f"• {row['Metric']}: p = {row['P-Value']:.4f}")
                        
                        with col2:
                            st.markdown("**Least Significant Metrics:**")
                            least_significant = mcpt_summary.nlargest(3, 'P-Value')
                            for _, row in least_significant.iterrows():
                                st.write(f"• {row['Metric']}: p = {row['P-Value']:.4f}")
                    else:
                        st.info("MCPT analysis completed but no results available for display.")
                else:
                    st.info("MCPT significance testing is disabled or failed. Enable in backtester configuration to see statistical significance analysis.")
                
                # Bootstrap Confidence Intervals Results
                if 'bootstrap_results' in results and 'summary' in results['bootstrap_results']:
                    st.subheader("Bootstrap Confidence Intervals")
                    st.markdown("**Bootstrap Analysis** - Confidence intervals computed by resampling trade P&L with replacement")
                    
                    bootstrap_summary = results['bootstrap_results']['summary']
                    if not bootstrap_summary.empty:
                        # Display Bootstrap results in a clean table
                        st.dataframe(
                            bootstrap_summary.style.format({
                                'Observed': '{:.4f}',
                                'CI Lower': '{:.4f}',
                                'CI Upper': '{:.4f}',
                                'CI Width': '{:.4f}',
                                'Bootstrap Mean': '{:.4f}',
                                'Bootstrap Std': '{:.4f}'
                            }),
                            width='stretch'
                        )
                        
                        # Confidence interval summary
                        st.subheader("Confidence Interval Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_ci_width = bootstrap_summary['CI Width'].mean()
                            st.metric("Average CI Width", f"{avg_ci_width:.4f}")
                        
                        with col2:
                            confidence_level = results['bootstrap_results']['config']['confidence_level']
                            st.metric("Confidence Level", f"{confidence_level:.1%}")
                        
                        with col3:
                            n_samples = results['bootstrap_results']['config']['n_bootstrap']
                            st.metric("Bootstrap Samples", f"{n_samples:,}")
                        
                        # Bootstrap histograms
                        if 'plots' in results['bootstrap_results']:
                            st.subheader("Bootstrap Distribution Histograms")
                            plots = results['bootstrap_results']['plots']
                            
                            # Display key metrics histograms
                            key_metrics = ['sharpe_ratio', 'cagr', 'max_drawdown', 'total_return']
                            available_plots = {k: v for k, v in plots.items() if k in key_metrics}
                            
                            if available_plots:
                                # Create tabs for different metrics
                                metric_tabs = st.tabs([metric.replace('_', ' ').title() for metric in available_plots.keys()])
                                
                                for i, (metric_name, plot) in enumerate(available_plots.items()):
                                    with metric_tabs[i]:
                                        st.plotly_chart(plot, width='stretch')
                                        
                                        # Add interpretation
                                        result = results['bootstrap_results']['results'].get(metric_name)
                                        if result:
                                            st.markdown(f"""
                                            **Interpretation for {metric_name.replace('_', ' ').title()}:**
                                            - **Observed Value**: {result.observed_value:.4f}
                                            - **95% Confidence Interval**: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]
                                            - **Bootstrap Mean**: {result.mean_bootstrap:.4f}
                                            - **Bootstrap Std**: {result.std_bootstrap:.4f}
                                            """)
                    else:
                        st.info("Bootstrap analysis completed but no results available for display.")
                else:
                    st.info("Bootstrap confidence intervals are disabled or failed. Enable in backtester configuration to see confidence interval analysis.")
            
                # Charts section with sub-tabs
                st.subheader("Performance Analysis")
                
                # Create sub-tabs for different analysis views
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Performance Charts", "Significance Analysis", "Trade Analysis"])
                
                with chart_tab1:
                    col1, col2 = st.columns([2, 1])
                
                    with col1:
                        st.subheader("Equity Curve")
                        if isinstance(equity, pd.DataFrame):
                            st.line_chart(equity['equity'])
                        else:
                            st.line_chart(equity)
                    
                    with col2:
                        st.subheader("All Metrics")
                        for key, value in metrics.items():
                            if key not in ['equity_curve', 'trades'] and isinstance(value, (int, float)):
                                st.metric(
                                    key.replace('_', ' ').title(),
                                    f"{value:.2%}" if 'return' in key or 'drawdown' in key else f"{value:.2f}" if isinstance(value, float) else value
                                )
                
                with chart_tab2:
                    st.subheader("Statistical Significance Analysis")
                    st.markdown("**Observed vs Null Distributions** - Compare your strategy's performance against random chance")
                    
                    # Check if we have MCPT results
                    if 'mcpt_results' in results and 'results' in results['mcpt_results']:
                        mcpt_results = results['mcpt_results']['results']
                        
                        # Key metrics to display
                        key_metrics = ['sharpe_ratio', 'total_return', 'cagr', 'max_drawdown']
                        available_metrics = [m for m in key_metrics if m in mcpt_results]
                        
                        if available_metrics:
                            # Create columns for side-by-side histograms
                            if len(available_metrics) >= 2:
                                col1, col2 = st.columns(2)
                                cols = [col1, col2]
                            else:
                                cols = [st.container()] * len(available_metrics)
                            
                            for i, metric_name in enumerate(available_metrics):
                                with cols[i % len(cols)]:
                                    mcpt_result = mcpt_results[metric_name]
                                    
                                    # Create histogram
                                    fig = go.Figure()
                                    
                                    # Add null distribution histogram
                                    fig.add_trace(go.Histogram(
                                        x=mcpt_result.null_values,
                                        nbinsx=50,
                                        name=f'Null Distribution ({metric_name.replace("_", " ").title()})',
                                        opacity=0.7,
                                        marker_color='lightblue'
                                    ))
                                    
                                    # Add observed value as vertical line
                                    fig.add_vline(
                                        x=mcpt_result.observed_value,
                                        line_dash="dash",
                                        line_color="red",
                                        line_width=3,
                                        annotation_text=f"Observed: {mcpt_result.observed_value:.4f}",
                                        annotation_position="top"
                                    )
                                    
                                    # Add confidence interval lines
                                    fig.add_vline(
                                        x=mcpt_result.confidence_interval[0],
                                        line_dash="dot",
                                        line_color="green",
                                        line_width=2,
                                        annotation_text=f"CI Lower: {mcpt_result.confidence_interval[0]:.4f}",
                                        annotation_position="bottom"
                                    )
                                    
                                    fig.add_vline(
                                        x=mcpt_result.confidence_interval[1],
                                        line_dash="dot",
                                        line_color="green",
                                        line_width=2,
                                        annotation_text=f"CI Upper: {mcpt_result.confidence_interval[1]:.4f}",
                                        annotation_position="bottom"
                                    )
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title=f'{metric_name.replace("_", " ").title()} - Observed vs Null',
                                        xaxis_title=metric_name.replace("_", " ").title(),
                                        yaxis_title='Frequency',
                                        showlegend=False,
                                        height=400,
                                        template='plotly_white'
                                    )
                                    
                                    st.plotly_chart(fig, width='stretch')
                                    
                                    # Display p-value and significance
                                    col_p1, col_p2, col_p3 = st.columns(3)
                                    with col_p1:
                                        st.metric("P-Value", f"{mcpt_result.p_value:.4f}")
                                    with col_p2:
                                        significance_color = "🟢" if mcpt_result.is_significant else "🔴"
                                        st.metric("Significant", f"{significance_color} {'Yes' if mcpt_result.is_significant else 'No'}")
                                    with col_p3:
                                        st.metric("Confidence Level", f"{mcpt_result.confidence_level:.1%}")
                        else:
                            st.info("No MCPT results available for significance analysis.")
                    else:
                        st.info("MCPT significance testing is disabled or failed. Enable in backtester configuration to see significance analysis.")
                
                with chart_tab3:
                    # Enhanced trades section
                    st.subheader("Trade Analysis")
                    
                    # Convert trades to DataFrame if needed
                    if isinstance(trades, list) and trades:
                        trades_df = pd.DataFrame(trades)
                    elif isinstance(trades, pd.DataFrame):
                        trades_df = trades
                    else:
                        trades_df = pd.DataFrame()
                    
                    if not trades_df.empty:
                        # Trade P&L visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'pnl' in trades_df.columns:
                                pnl_chart = create_trade_pnl_chart(trades_df)
                                if pnl_chart:
                                    st.plotly_chart(pnl_chart, width='stretch')
                        
                        with col2:
                            if 'entry_time' in trades_df.columns and 'pnl' in trades_df.columns:
                                heatmap = create_trade_heatmap(trades_df)
                                if heatmap:
                                    st.plotly_chart(heatmap, width='stretch')
                    
                        # Collapsible trade log
                        with st.expander(f"Trade Log ({len(trades_df)} trades)", expanded=False):
                            # Add P&L styling to the dataframe
                            if 'pnl' in trades_df.columns:
                                def style_pnl(val):
                                    if val > 0:
                                        return 'color: #4CAF50; font-weight: bold'
                                    elif val < 0:
                                        return 'color: #f44336; font-weight: bold'
                                    else:
                                        return ''
                                
                                styled_trades = trades_df.style.map(style_pnl, subset=['pnl'])
                                st.dataframe(styled_trades, width='stretch')
                            else:
                                st.dataframe(trades_df, width='stretch')
                        
                        # Export buttons
                        st.subheader("Export Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("Export Trades CSV"):
                                csv = trades_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Trades CSV",
                                    data=csv,
                                    file_name=f"trades_{'-'.join(selected_tickers)}_{start}_{end}.csv",
                                    mime="text/csv"
                                )
                        
                        with col2:
                            if st.button("Export Metrics CSV"):
                                metrics_df = pd.DataFrame([metrics])
                                csv = metrics_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Metrics CSV",
                                    data=csv,
                                    file_name=f"metrics_{'-'.join(selected_tickers)}_{start}_{end}.csv",
                                    mime="text/csv"
                                )
                        
                        with col3:
                            if st.button("Export Equity Curve CSV"):
                                if isinstance(equity, pd.DataFrame):
                                    csv = equity.to_csv()
                                else:
                                    csv = equity.to_frame().to_csv()
                                st.download_button(
                                    label="Download Equity CSV",
                                    data=csv,
                                    file_name=f"equity_{'-'.join(selected_tickers)}_{start}_{end}.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.info("No trades executed during this period.")
                
                # Per-ticker results for portfolio strategies
                if is_portfolio_strategy and 'symbol_performance' in metrics:
                    st.subheader("Per-Ticker Performance")
                    
                    # Create tabs for each ticker
                    ticker_tabs = st.tabs(selected_tickers)
                    
                    for i, ticker in enumerate(selected_tickers):
                        with ticker_tabs[i]:
                            if ticker in metrics['symbol_performance']:
                                ticker_perf = metrics['symbol_performance'][ticker]
                                
                                # Ticker metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("P&L", f"${ticker_perf.get('pnl', 0):.2f}")
                                with col2:
                                    st.metric("Trades", ticker_perf.get('trades', 0))
                                with col3:
                                    st.metric("Avg Return", f"{ticker_perf.get('avg_return', 0):.2%}")
                                with col4:
                                    st.metric("Symbol", ticker)
                            else:
                                st.info(f"No trades executed for {ticker}")
                
                # MLflow link
                st.markdown(f"[Open in MLflow]({run_url})")
            
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
    else:
        st.info("Configure parameters in the sidebar and click 'Run Backtest' to begin.")

with tab2:
    st.header("Walk-Forward Analysis")
    st.markdown("Analyze strategy performance and statistical significance over rolling time windows.")
    
    # Walk-forward configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window_length = st.number_input("Window Length (days)", 60, 1000, 252, step=21,
                                       help="Length of each rolling window in trading days")
    
    with col2:
        step_size = st.number_input("Step Size (days)", 5, 100, 21, step=5,
                                   help="Step size between windows in trading days")
    
    with col3:
        min_trades = st.number_input("Minimum Trades", 5, 50, 10, step=5,
                                    help="Minimum trades required per window for analysis")
    
    # MCPT configuration for walk-forward
    st.subheader("Statistical Testing Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        wf_mcpt_permutations = st.number_input("MCPT Permutations", 100, 2000, 500, step=100,
                                              help="Number of permutations for each window (smaller for speed)")
    
    with col2:
        wf_significance_level = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01,
                                         help="P-value threshold for significance")
    
    # Run walk-forward analysis button
    run_wf_btn = st.button("Run Walk-Forward Analysis", width='stretch')
    
    if run_wf_btn:
        with st.spinner("Running walk-forward analysis... This may take a few minutes."):
            try:
                # Load data for analysis
                if len(selected_tickers) > 1:
                    st.warning("Walk-forward analysis works best with single ticker. Using first ticker only.")
                
                ticker = selected_tickers[0]
                df = load_yf_data([ticker], str(start), str(end))
                
                if df.empty:
                    st.error("No data available for walk-forward analysis")
                else:
                    # Prepare data for single-ticker strategy by renaming columns
                    if f"{ticker}_close" in df.columns:
                        df = df.rename(columns={
                            f"{ticker}_close": "close",
                            f"{ticker}_open": "open",
                            f"{ticker}_high": "high",
                            f"{ticker}_low": "low",
                            f"{ticker}_volume": "volume"
                        })
                    
                    # Create strategy
                    strat = get_strategy(strategy_name, **strategy_params)
                    
                    # Create backtester
                    bt = Backtester(
                        commission=fee_bps/10000, 
                        slippage=slip_bps/10000,
                        enable_mcpt=False,  # We'll run MCPT in walk-forward
                        enable_bootstrap=False  # We'll run Bootstrap in walk-forward
                    )
                    
                    # Create walk-forward config
                    wf_config = WalkForwardConfig(
                        window_length=window_length,
                        step_size=step_size,
                        min_trades=min_trades,
                        significance_level=wf_significance_level
                    )
                    
                    # Run walk-forward analysis
                    wf_analyzer = WalkForwardAnalyzer(wf_config)
                    wf_results = wf_analyzer.analyze_strategy(df, strat, bt, str(start), str(end))
                    
                    if 'error' in wf_results:
                        st.error(f"Walk-forward analysis failed: {wf_results['error']}")
                    else:
                        st.success("Walk-forward analysis completed successfully!")
                        
                        # Display summary
                        st.subheader("Analysis Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Windows", len(wf_results['windows']))
                        
                        with col2:
                            avg_sharpe = wf_results['rolling_metrics']['sharpe_ratio'].mean()
                            st.metric("Average Sharpe", f"{avg_sharpe:.3f}")
                        
                        with col3:
                            significant_windows = wf_results['rolling_pvalues']['sharpe_ratio_significant'].sum()
                            total_windows = len(wf_results['rolling_pvalues'])
                            st.metric("Significant Windows", f"{significant_windows}/{total_windows}")
                        
                        with col4:
                            significance_rate = significant_windows / total_windows if total_windows > 0 else 0
                            st.metric("Significance Rate", f"{significance_rate:.1%}")
                        
                        # Display visualizations
                        st.subheader("Walk-Forward Visualizations")
                        
                        plots = wf_results.get('plots', {})
                        
                        if 'rolling_sharpe' in plots:
                            st.plotly_chart(plots['rolling_sharpe'], width='stretch')
                        
                        # Add rolling Sharpe with significance regions
                        if 'rolling_metrics' in wf_results and 'rolling_pvalues' in wf_results:
                            st.subheader("Rolling Sharpe with Significance Regions")
                            
                            # Create enhanced rolling Sharpe plot with significance shading
                            fig = go.Figure()
                            
                            # Add rolling Sharpe line
                            fig.add_trace(go.Scatter(
                                x=wf_results['rolling_metrics']['end_date'],
                                y=wf_results['rolling_metrics']['sharpe_ratio'],
                                mode='lines+markers',
                                name='Rolling Sharpe Ratio',
                                line=dict(color='blue', width=2),
                                marker=dict(size=6)
                            ))
                            
                            # Add significance regions
                            if 'sharpe_ratio_significant' in wf_results['rolling_pvalues'].columns:
                                significant_data = wf_results['rolling_pvalues']['sharpe_ratio_significant']
                                dates = wf_results['rolling_metrics']['end_date']
                                
                                # Create shaded regions for significant periods
                                in_significant_region = False
                                start_date = None
                                
                                for i, (date, is_significant) in enumerate(zip(dates, significant_data)):
                                    if is_significant and not in_significant_region:
                                        # Start of significant region
                                        start_date = date
                                        in_significant_region = True
                                    elif not is_significant and in_significant_region:
                                        # End of significant region
                                        fig.add_vrect(
                                            x0=start_date,
                                            x1=date,
                                            fillcolor="green",
                                            opacity=0.2,
                                            layer="below",
                                            line_width=0,
                                            annotation_text="Significant Period",
                                            annotation_position="top left"
                                        )
                                        in_significant_region = False
                                
                                # Handle case where significant region extends to end
                                if in_significant_region:
                                    fig.add_vrect(
                                        x0=start_date,
                                        x1=dates.iloc[-1],
                                        fillcolor="green",
                                        opacity=0.2,
                                        layer="below",
                                        line_width=0,
                                        annotation_text="Significant Period",
                                        annotation_position="top left"
                                    )
                            
                            # Add zero line
                            fig.add_hline(
                                y=0, 
                                line_dash="dash", 
                                line_color="gray",
                                annotation_text="Zero Line"
                            )
                            
                            # Update layout
                            fig.update_layout(
                                title='Rolling Sharpe Ratio with Statistical Significance Regions',
                                xaxis_title='Date',
                                yaxis_title='Sharpe Ratio',
                                height=500,
                                showlegend=True,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, width='stretch')
                            
                            # Add significance summary
                            if 'sharpe_ratio_significant' in wf_results['rolling_pvalues'].columns:
                                significant_count = wf_results['rolling_pvalues']['sharpe_ratio_significant'].sum()
                                total_count = len(wf_results['rolling_pvalues'])
                                significance_rate = significant_count / total_count if total_count > 0 else 0
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Significant Windows", f"{significant_count}/{total_count}")
                                with col2:
                                    st.metric("Significance Rate", f"{significance_rate:.1%}")
                                with col3:
                                    avg_pvalue = wf_results['rolling_pvalues']['sharpe_ratio_pvalue'].mean()
                                    st.metric("Average P-Value", f"{avg_pvalue:.4f}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'significance_heatmap' in plots:
                                st.plotly_chart(plots['significance_heatmap'], width='stretch')
                        
                        with col2:
                            if 'pvalues_strip' in plots:
                                st.plotly_chart(plots['pvalues_strip'], width='stretch')
                        
                        if 'performance_significance' in plots:
                            st.plotly_chart(plots['performance_significance'], width='stretch')
                        
                        # Display detailed results table
                        st.subheader("Detailed Results")
                        summary_df = wf_analyzer.get_summary()
                        if not summary_df.empty:
                            st.dataframe(summary_df, width='stretch')
                        
                        # Store results for AI assistant
                        st.session_state.walkforward_results = wf_results
                        
            except Exception as e:
                st.error(f"Error running walk-forward analysis: {e}")

with tab3:
    st.header("Strategy Library")
    st.write("Available trading strategies in Neural Quant:")
    
    for strategy_name in strategy_registry.list_strategies():
        with st.expander(f"{strategy_name.replace('_', ' ').title()}", expanded=False):
            strategy_info = strategy_registry.get_strategy_info(strategy_name)
            st.write(f"**Description:** {strategy_info.get('description', 'No description available')}")
            st.write(f"**Type:** {'Portfolio Strategy' if strategy_info.get('is_portfolio_strategy', False) else 'Single Ticker Strategy'}")
            
            if strategy_info.get('parameters'):
                st.write("**Parameters:**")
                for param_name, param_info in strategy_info['parameters'].items():
                    default_val = param_info.get('default', 'No default')
                    param_type = param_info.get('annotation', 'Unknown type')
                    st.write(f"- {param_name}: {param_type} (default: {default_val})")

with tab4:
    st.header("Metrics Key")
    st.write("Understanding the performance metrics used in Neural Quant:")
    
    metrics_info = {
        "Total Return": {
            "definition": "The total percentage return over the entire backtest period",
            "calculation": "(Final Value - Initial Value) / Initial Value",
            "benchmark_range": "Varies by market conditions. Positive is good, negative is bad.",
            "good_threshold": "> 10%",
            "bad_threshold": "< -10%"
        },
        "Sharpe Ratio": {
            "definition": "Risk-adjusted return measure. Higher values indicate better risk-adjusted performance",
            "calculation": "(Portfolio Return - Risk-free Rate) / Portfolio Volatility",
            "benchmark_range": "0.5-2.0 is typical for good strategies",
            "good_threshold": "> 1.0",
            "bad_threshold": "< 0.5"
        },
        "Max Drawdown": {
            "definition": "The maximum peak-to-trough decline in portfolio value",
            "calculation": "Max((Peak - Trough) / Peak)",
            "benchmark_range": "Should be manageable for risk tolerance",
            "good_threshold": "> -10%",
            "bad_threshold": "< -20%"
        },
        "Volatility": {
            "definition": "Standard deviation of returns, measuring price variability",
            "calculation": "Standard deviation of daily returns * sqrt(252)",
            "benchmark_range": "10-30% is typical for equity strategies",
            "good_threshold": "< 20%",
            "bad_threshold": "> 30%"
        },
        "Win Rate": {
            "definition": "Percentage of profitable trades",
            "calculation": "Number of profitable trades / Total trades",
            "benchmark_range": "40-60% is typical",
            "good_threshold": "> 50%",
            "bad_threshold": "< 40%"
        },
        "Profit Factor": {
            "definition": "Ratio of gross profit to gross loss",
            "calculation": "Total profit / Total loss",
            "benchmark_range": "1.0+ indicates profitable strategy",
            "good_threshold": "> 1.5",
            "bad_threshold": "< 1.0"
        },
        "Sortino Ratio": {
            "definition": "Sharpe ratio using only downside deviation",
            "calculation": "(Portfolio Return - Risk-free Rate) / Downside Deviation",
            "benchmark_range": "Similar to Sharpe but focuses on downside risk",
            "good_threshold": "> 1.0",
            "bad_threshold": "< 0.5"
        },
        "Calmar Ratio": {
            "definition": "Annual return divided by maximum drawdown",
            "calculation": "Annual Return / Max Drawdown",
            "benchmark_range": "Higher is better, 1.0+ is good",
            "good_threshold": "> 1.0",
            "bad_threshold": "< 0.5"
        }
    }
    
    for metric_name, info in metrics_info.items():
        with st.expander(f"{metric_name}", expanded=False):
            st.write(f"**Definition:** {info['definition']}")
            st.write(f"**Calculation:** {info['calculation']}")
            st.write(f"**Benchmark Range:** {info['benchmark_range']}")
            st.write(f"**Good Threshold:** {info['good_threshold']}")
            st.write(f"**Bad Threshold:** {info['bad_threshold']}")
    
    st.markdown("---")
    
    # Statistical Analysis Section
    st.header("Statistical Analysis Concepts")
    st.markdown("Understanding the statistical methods used to validate trading strategies:")
    
    with st.expander("P-Values and Statistical Significance", expanded=True):
        st.markdown("""
        **What are P-values?**
        
        A p-value tells you the probability that your strategy's performance could have occurred by random chance alone. 
        It's a measure of statistical significance that helps distinguish between genuine skill and luck.
        
        **How to interpret P-values:**
        - **P < 0.05 (5%)**: Your strategy is statistically significant - there's less than 5% chance the results are due to luck
        - **P < 0.01 (1%)**: Highly significant - less than 1% chance of random occurrence
        - **P > 0.05**: Not statistically significant - results could be due to random chance
        
        **Why this matters for trading:**
        - **Reduces Overfitting Risk**: Statistical significance helps ensure your strategy isn't just fitting to noise
        - **Validates Strategy Logic**: Significant results suggest your strategy has genuine predictive power
        - **Prevents False Confidence**: Non-significant results warn you that performance might not persist
        
        **Example**: If your strategy has a Sharpe ratio of 1.5 with p-value of 0.02, there's only a 2% chance 
        that a random strategy would achieve this performance. This gives you confidence the strategy has real skill.
        """)
    
    with st.expander("Bootstrap Confidence Intervals", expanded=True):
        st.markdown("""
        **What is Bootstrapping?**
        
        Bootstrapping is a statistical method that resamples your trade data with replacement to estimate 
        the uncertainty around your performance metrics. It helps answer: "How confident can I be in these results?"
        
        **How Bootstrapping Works:**
        1. **Resample Trades**: Randomly select trades from your history (allowing duplicates)
        2. **Calculate Metrics**: Compute Sharpe ratio, returns, etc. for this resampled set
        3. **Repeat Many Times**: Do this 1000+ times to build a distribution
        4. **Find Confidence Intervals**: Determine the range where 95% of results fall
        
        **Why Bootstrapping Helps:**
        - **Robustness Check**: Shows how sensitive your results are to specific trades
        - **Uncertainty Quantification**: Provides confidence intervals for all metrics
        - **Risk Assessment**: Helps understand the range of possible outcomes
        - **Validation**: Confirms your strategy isn't dependent on a few lucky trades
        
        **Example**: If your Sharpe ratio is 1.2 with 95% confidence interval [0.8, 1.6], 
        you can be 95% confident the true Sharpe ratio falls within this range.
        """)
    
    with st.expander("Monte Carlo Permutation Testing (MCPT)", expanded=True):
        st.markdown("""
        **What is MCPT?**
        
        Monte Carlo Permutation Testing shuffles your returns or signals randomly to create "null" distributions 
        that represent what random chance would produce. It's the gold standard for testing statistical significance.
        
        **How MCPT Works:**
        1. **Shuffle Data**: Randomly reorder your returns or trading signals
        2. **Calculate Metrics**: Compute performance metrics for shuffled data
        3. **Repeat Many Times**: Create thousands of random permutations
        4. **Compare Results**: See where your actual performance ranks against random chance
        
        **Why MCPT is Powerful:**
        - **Null Hypothesis Testing**: Directly tests if your results could be due to chance
        - **Preserves Data Structure**: Maintains the statistical properties of your returns
        - **Comprehensive Testing**: Tests multiple metrics simultaneously
        - **Overfitting Protection**: Helps prevent strategies that only work on historical data
        
        **Example**: If your strategy's Sharpe ratio of 1.5 ranks in the top 2% of 1000 random permutations, 
        there's only a 2% chance this performance is due to luck (p-value = 0.02).
        """)
    
    with st.expander("Walk-Forward Analysis", expanded=True):
        st.markdown("""
        **What is Walk-Forward Analysis?**
        
        Walk-forward analysis tests your strategy across rolling time windows to see how performance 
        and statistical significance change over time. It's crucial for understanding strategy stability.
        
        **How Walk-Forward Works:**
        1. **Rolling Windows**: Test strategy on overlapping time periods (e.g., 1-year windows)
        2. **Statistical Testing**: Run MCPT on each window to check significance
        3. **Track Evolution**: See how performance and significance change over time
        4. **Identify Regimes**: Spot when strategy works vs. when it fails
        
        **Why Walk-Forward Matters:**
        - **Temporal Stability**: Ensures strategy works across different market conditions
        - **Regime Detection**: Identifies when strategy is most/least effective
        - **Overfitting Prevention**: Tests strategy on out-of-sample data
        - **Real-World Validation**: Mimics how you'd actually use the strategy
        
        **Example**: A strategy might show high significance in 2020-2021 but lose significance in 2022, 
        indicating it may not work in all market conditions.
        """)
    
    with st.expander("Reducing Overfitting Risk", expanded=True):
        st.markdown("""
        **What is Overfitting?**
        
        Overfitting occurs when a strategy performs well on historical data but fails in live trading. 
        It's like memorizing answers to a test instead of learning the underlying concepts.
        
        **How Statistical Testing Reduces Overfitting:**
        
        **1. Significance Testing**
        - **P-values < 0.05**: Indicates genuine skill, not just curve-fitting
        - **Multiple Metrics**: Tests several performance measures simultaneously
        - **Robust Validation**: Ensures results aren't due to random chance
        
        **2. Bootstrap Analysis**
        - **Confidence Intervals**: Shows uncertainty around performance estimates
        - **Sensitivity Testing**: Reveals how dependent results are on specific trades
        - **Risk Assessment**: Quantifies the range of possible outcomes
        
        **3. Walk-Forward Testing**
        - **Out-of-Sample Validation**: Tests strategy on unseen data
        - **Temporal Stability**: Ensures consistent performance over time
        - **Regime Analysis**: Identifies when strategy works vs. fails
        
        **Best Practices:**
        - **Always Test Significance**: Never rely solely on backtest performance
        - **Use Multiple Methods**: Combine MCPT, Bootstrap, and Walk-Forward
        - **Be Conservative**: Prefer strategies with strong statistical evidence
        - **Monitor Over Time**: Continuously validate strategy performance
        """)
    
    st.markdown("---")
    st.write("**Note:** These statistical methods are essential for building robust, reliable trading strategies. Always validate your strategies with proper statistical testing before deploying capital.")

with tab5:
    st.header("Market Watch")
    st.write("Live price charts for all supported tickers")
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    # Define the 12 tickers by asset class
    market_tickers = {
        "Equities": ["AAPL", "TSLA", "NVDA", "PLTR", "GOOG", "LMT", "NOC"],
        "Crypto": ["BTC-USD", "SOL-USD"],
        "Commodities": ["CL=F"],
        "FX/Metals": ["GBPUSD=X", "XAUUSD=X"]
    }
    
    # Flatten for easy access
    all_tickers = [ticker for tickers in market_tickers.values() for ticker in tickers]
    
    # Fallback symbols
    fallback_symbols = {
        "XAUUSD=X": "GC=F"  # Gold fallback
    }
    
    # Define asset class intervals
    asset_intervals = {
        "crypto": "1d",  # BTC-USD, SOL-USD (use daily for longer periods)
        "equities": "1d",
        "commodities": "1d",  # CL=F
        "fx_metals": "1h"  # GBPUSD=X, XAUUSD=X
    }
    
    def get_asset_class(ticker):
        """Determine asset class for ticker."""
        if ticker in ["BTC-USD", "SOL-USD"]:
            return "crypto"
        elif ticker in ["AAPL", "TSLA", "NVDA", "PLTR", "GOOG", "LMT", "NOC"]:
            return "equities"
        elif ticker in ["CL=F"]:
            return "commodities"
        elif ticker in ["GBPUSD=X", "XAUUSD=X"]:
            return "fx_metals"
        else:
            return "equities"  # Default fallback
    
    # Market Watch controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        selected_ticker = st.selectbox(
            "Select ticker for detailed view:",
            options=all_tickers,
            index=0,
            help="Choose a ticker to view detailed chart"
        )
    
    with col2:
        period_presets = {
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "YTD": "ytd",
            "1Y": 365
        }
        selected_period = st.selectbox(
            "Time period:",
            options=list(period_presets.keys()),
            index=1,  # Default to 3M
            help="Select time period for data"
        )
        days_back = period_presets[selected_period]
    
    with col3:
        chart_mode = st.selectbox(
            "Chart mode:",
            options=["Price", "Normalised (%)"],
            index=0,
            help="Price: raw values, Normalised: performance since first point"
        )
    
    with col4:
        refresh_data = st.button("Refresh All Data", width='stretch')
    
    # Force refresh for selected ticker
    if selected_ticker:
        col1, col2 = st.columns([3, 1])
        with col2:
            force_refresh = st.button("Force Refresh Selected", width='stretch')
            if force_refresh:
                # Clear cache for this specific ticker
                if 'market_data' in st.session_state:
                    # Remove from cache and reload
                    del st.session_state.market_data
                    del st.session_state.price_info
                    st.rerun()
    
    
    def get_current_price_and_source(data, ticker, asset_class):
        """Get current price with source information."""
        if data.empty:
            return None, None, None, None
        
        # Get the close price column
        close_col = f"{ticker}_close"
        if close_col not in data.columns:
            # Try alternative column names
            for col in data.columns:
                if 'close' in col.lower():
                    close_col = col
                    break
            else:
                return None, None, None, None
        
        price_data = data[close_col].dropna()
        if price_data.empty:
            return None, None, None, None
        
        # Get current price with source preference
        current_price = None
        source = None
        timestamp = None
        is_stale = False
        
        # Try live/intraday first for crypto (only for very recent data)
        if asset_class == "crypto":
            try:
                # Try 1-minute data for most recent price (last 2 days only)
                intraday_data = load_yf_data([ticker], days_back=2, timeframe='1m')
                if not intraday_data.empty:
                    intraday_close = f"{ticker}_close"
                    if intraday_close in intraday_data.columns:
                        intraday_prices = intraday_data[intraday_close].dropna()
                        if not intraday_prices.empty:
                            current_price = intraday_prices.iloc[-1]
                            source = "1m intraday"
                            timestamp = intraday_prices.index[-1]
                            
                            # Check if stale (older than 3x interval = 3 minutes)
                            if (pd.Timestamp.now() - timestamp).total_seconds() > 180:
                                is_stale = True
            except:
                pass
        
        # Fallback to daily close
        if current_price is None:
            current_price = price_data.iloc[-1]
            source = "daily close"
            timestamp = price_data.index[-1]
            
            # Check if stale (older than 3x interval)
            interval_minutes = {"1d": 1440, "1h": 60, "5m": 5, "1m": 1}
            interval = asset_intervals.get(asset_class, "1d")
            stale_threshold = interval_minutes.get(interval, 1440) * 3
            
            if (pd.Timestamp.now() - timestamp).total_seconds() > stale_threshold * 60:
                is_stale = True
        
        return current_price, source, timestamp, is_stale
    
    # Load market data with proper error handling and fallbacks
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_market_data(tickers, days_back):
        """Load market data for all tickers with fallbacks and proper intervals."""
        data_dict = {}
        price_info = {}
        
        for ticker in tickers:
            try:
                asset_class = get_asset_class(ticker)
                interval = asset_intervals[asset_class]
                
                # Try primary symbol first
                df = load_yf_data([ticker], days_back=days_back, timeframe=interval)
                actual_ticker = ticker
                
                if df.empty and ticker in fallback_symbols:
                    # Try fallback symbol
                    fallback = fallback_symbols[ticker]
                    df = load_yf_data([fallback], days_back=days_back, timeframe=interval)
                    if not df.empty:
                        actual_ticker = fallback  # Use fallback symbol
                
                if not df.empty:
                    # Get the close price column
                    close_col = f"{actual_ticker}_close"
                    if close_col in df.columns:
                        price_data = df[close_col].dropna()
                    else:
                        # Try alternative column names
                        for col in df.columns:
                            if 'close' in col.lower():
                                price_data = df[col].dropna()
                                break
                        else:
                            continue
                    
                    # Remove timezone and sort by date
                    price_data.index = price_data.index.tz_localize(None) if price_data.index.tz else price_data.index
                    price_data = price_data.sort_index()
                    
                    # Get current price and source info
                    current_price, source, timestamp, is_stale = get_current_price_and_source(
                        df, actual_ticker, asset_class
                    )
                    
                    data_dict[actual_ticker] = price_data
                    price_info[actual_ticker] = {
                        'current_price': current_price,
                        'source': source,
                        'timestamp': timestamp,
                        'is_stale': is_stale,
                        'asset_class': asset_class,
                        'interval': interval,
                        'original_ticker': ticker,
                        'actual_ticker': actual_ticker
                    }
                    
            except Exception as e:
                # Silently continue for failed tickers
                continue
        
        return data_dict, price_info
    
    # Load data
    if refresh_data or 'market_data' not in st.session_state or st.session_state.get('selected_period') != selected_period:
        with st.spinner("Loading market data..."):
            market_data, price_info = load_market_data(all_tickers, days_back)
            st.session_state.market_data = market_data
            st.session_state.price_info = price_info
            st.session_state.selected_period = selected_period
    else:
        market_data = st.session_state.get('market_data', {})
        price_info = st.session_state.get('price_info', {})
    
    if market_data:
        # Display grid of small charts
        st.subheader("Market Overview")
        
        # Create 4 columns for the grid
        cols = st.columns(4)
        
        for i, ticker in enumerate(all_tickers):
            with cols[i % 4]:
                # Find the actual ticker (may be fallback)
                actual_ticker = None
                for actual, info in price_info.items():
                    if info['original_ticker'] == ticker:
                        actual_ticker = actual
                        break
                
                if actual_ticker and actual_ticker in market_data:
                    try:
                        price_data = market_data[actual_ticker]
                        ticker_info = price_info[actual_ticker]
                        
                        if not price_data.empty and ticker_info['current_price'] is not None:
                            # Get current price and change
                            current_price = ticker_info['current_price']
                            prev_price = price_data.iloc[-2] if len(price_data) > 1 else current_price
                            change = current_price - prev_price
                            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                            
                            # Format timestamp
                            timestamp = ticker_info['timestamp']
                            if timestamp:
                                time_str = timestamp.strftime("%H:%M:%S")
                                date_str = timestamp.strftime("%m/%d")
                            else:
                                time_str = "N/A"
                                date_str = "N/A"
                            
                            # Check for price mismatch
                            last_chart_price = price_data.iloc[-1]
                            price_mismatch = abs(current_price - last_chart_price) / current_price > 0.01
                            
                            # Create ticker box with enhanced information
                            stale_badge = " [STALE]" if ticker_info['is_stale'] else ""
                            mismatch_warning = " [MISMATCH]" if price_mismatch else ""
                            fallback_indicator = " (Fallback)" if actual_ticker != ticker else ""
                            
                            # Create ticker box using Streamlit components instead of raw HTML
                            with st.container():
                                st.markdown(f"**{ticker}{fallback_indicator}**{stale_badge}{mismatch_warning}")
                                st.metric("Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
                                st.caption(f"{ticker_info['source']} • {time_str}")
                                st.caption(f"{actual_ticker} • {ticker_info['interval']}")
                            
                            # Small line chart in a container
                            with st.container():
                                st.line_chart(price_data.tail(30), height=120)
                        else:
                            st.info("No data for this period")
                    except Exception as e:
                        st.info("No data for this period")
                else:
                    st.info("No data for this period")
        
        # Detailed chart for selected ticker
        st.markdown("---")
        st.subheader(f"Detailed Chart: {selected_ticker}")
        
        # Find the actual ticker (may be fallback)
        actual_ticker = None
        ticker_info = None
        for actual, info in price_info.items():
            if info['original_ticker'] == selected_ticker:
                actual_ticker = actual
                ticker_info = info
                break
        
        if actual_ticker and actual_ticker in market_data:
            price_data = market_data[actual_ticker]
            
            if not price_data.empty and ticker_info and ticker_info['current_price'] is not None:
                # Use the current price from price info, not the last chart point
                current_price = ticker_info['current_price']
                first_price = price_data.iloc[0]
                total_return = (current_price / first_price - 1) * 100
                
                # Calculate volatility from the exact series
                if len(price_data) > 1:
                    returns = price_data.pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5)  # Annualized
                else:
                    volatility = 0
                
                # Format timestamp
                timestamp = ticker_info['timestamp']
                if timestamp:
                    time_str = timestamp.strftime("%H:%M:%S")
                    date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = "N/A"
                    date_str = "N/A"
                
                # Display enhanced metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Total Return", f"{total_return:.2f}%")
                with col3:
                    st.metric("Volatility", f"{volatility:.2f}")
                with col4:
                    st.metric("Data Points", len(price_data))
                
                # Display source and timestamp information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Source", ticker_info['source'])
                with col2:
                    st.metric("Last Updated", time_str)
                with col3:
                    st.metric("Symbol", f"{actual_ticker} ({ticker_info['interval']})")
                
                # Show staleness warning if applicable
                if ticker_info['is_stale']:
                    st.warning(f"Data may be stale (last update: {date_str})")
                
                # Prepare data for plotting based on chart mode
                if chart_mode == "Normalised (%)":
                    # Normalize to percentage performance since first point
                    plot_data = ((price_data / first_price - 1) * 100)
                    y_axis_label = "Performance (%)"
                else:
                    # Raw price values
                    plot_data = price_data
                    y_axis_label = "Price ($)"
                
                # Large detailed chart with better spacing
                st.markdown(f"**{y_axis_label}**")
                st.line_chart(plot_data, height=350)
                
                # Price statistics in a more compact expander
                with st.expander("Price Statistics", expanded=False):
                    st.dataframe(price_data.describe(), width='stretch')
            else:
                st.warning("No data available for this ticker")
        else:
            st.warning("No data available for this ticker")
    else:
        st.error("No market data available. Click 'Refresh Data' to load prices.")
    
    # Market summary
    if market_data and price_info:
        st.subheader("Market Summary")
        
        # Calculate market-wide statistics from available data
        all_returns = []
        active_tickers = 0
        stale_tickers = 0
        
        for ticker, data in market_data.items():
            if len(data) > 1:
                returns = data.pct_change().dropna()
                all_returns.extend(returns.tolist())
                active_tickers += 1
                
                # Check for stale data
                if ticker in price_info and price_info[ticker]['is_stale']:
                    stale_tickers += 1
        
        if all_returns:
            import numpy as np
            market_volatility = np.std(all_returns) * (252 ** 0.5)
            avg_return = np.mean(all_returns) * 252  # Annualized
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Market Volatility", f"{market_volatility:.2f}")
            with col2:
                st.metric("Average Return", f"{avg_return:.2%}")
            with col3:
                st.metric("Active Tickers", active_tickers)
            with col4:
                st.metric("Stale Data", f"{stale_tickers}/{active_tickers}")
        
        # Show data freshness summary
        if stale_tickers > 0:
            st.warning(f"{stale_tickers} ticker(s) have stale data. Consider refreshing.")

# AI Chat Display
if st.session_state.chat_history:
    st.markdown("---")
    st.header("AI Assistant Chat")
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])