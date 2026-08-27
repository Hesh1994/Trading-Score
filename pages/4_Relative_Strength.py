"""
Relative Strength (Turnover-Based)
RS(t) = Stock Trading Value(t) / Benchmark Total Trading Value(t),
where Trading Value = Close x Volume. Reports RS per period, the Average RS
over the chosen interval, and whether relative participation is trending.
"""

import sys, os

_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import numpy as np
import datetime as _dt
import plotly.graph_objects as go

import relative_strength as rsm

st.set_page_config(
    page_title="Relative Strength",
    page_icon="⚖️",
    layout="wide",
)

# ── Page header ───────────────────────────────────────────────────────────────
_hdr_col, _btn_col = st.columns([4, 1])
with _hdr_col:
    st.title("⚖️ Relative Strength (Turnover)")
    st.caption(
        "**RS(t) = Stock Trading Value ÷ Benchmark Trading Value**, "
        "where Trading Value = Close × Volume. Measures how much of the market's "
        "money flow a stock is capturing, rather than how far its price has moved."
    )
_btn_col.markdown('<div style="margin-top: 1.6rem;"></div>', unsafe_allow_html=True)
_run = _btn_col.button("⚖️ Calculate RS", type="primary",
                       use_container_width=True, key="rs_run_btn")
st.markdown('<hr style="border:none;border-top:3px solid black;margin-top:0;margin-bottom:1rem;">',
            unsafe_allow_html=True)

# ── Sidebar: configuration ────────────────────────────────────────────────────
st.sidebar.header("⚙️ RS Configuration")

# Pull tickers already loaded by the other dashboards, if any
_known = list(dict.fromkeys(
    list(st.session_state.get('ta_ticker_list', [])) +
    list(st.session_state.get('canslim_ticker_list', []))
))

st.sidebar.subheader("🎯 Stock")
if _known:
    _src = st.sidebar.radio("Ticker source", ["From screener", "Type manually"],
                            horizontal=True, key="rs_src")
else:
    _src = "Type manually"

if _src == "From screener":
    ticker = st.sidebar.selectbox("Ticker", _known, key="rs_ticker_sel")
else:
    ticker = st.sidebar.text_input("Ticker", value="AAPL", key="rs_ticker_txt").strip().upper()

st.sidebar.subheader("📅 Data Range")
_c1, _c2 = st.sidebar.columns(2)
with _c1:
    start_date = st.date_input("Start Date",
                               value=_dt.date.today() - _dt.timedelta(days=180),
                               min_value=_dt.date(2010, 1, 1),
                               max_value=_dt.date.today(),
                               key="rs_start")
with _c2:
    end_date = st.date_input("End Date",
                             value=_dt.date.today(),
                             min_value=start_date,
                             max_value=_dt.date.today(),
                             key="rs_end")

st.sidebar.subheader("🏛️ Benchmark")
bench_mode = st.sidebar.selectbox(
    "Benchmark trading value from",
    ["ETF proxy", "Index constituents", "Official turnover (paste/upload)"],
    key="rs_bench_mode",
    help="ETF proxy is fastest but RS levels are only comparable to themselves. "
         "Constituents or official turnover give a true participation share.",
)

bench_proxy = None
bench_members = None
bench_turnover_df = None

if bench_mode == "ETF proxy":
    _presets = {
        "S&P 500 → SPY": ("SPY", "S&P 500"),
        "Nasdaq 100 → QQQ": ("QQQ", "Nasdaq 100"),
        "Dow 30 → DIA": ("DIA", "Dow 30"),
        "Russell 2000 → IWM": ("IWM", "Russell 2000"),
        "Saudi TASI → KSA": ("KSA", "Saudi TASI"),
        "Custom ETF…": (None, None),
    }
    _choice = st.sidebar.selectbox("Index", list(_presets.keys()), key="rs_proxy_preset")
    _p, _n = _presets[_choice]
    if _p is None:
        bench_proxy = st.sidebar.text_input("Proxy ETF ticker", value="SPY",
                                            key="rs_proxy_custom").strip().upper()
        bench_name = st.sidebar.text_input("Benchmark display name", value=bench_proxy,
                                           key="rs_proxy_name").strip() or bench_proxy
    else:
        bench_proxy, bench_name = _p, _n

elif bench_mode == "Index constituents":
    bench_name = st.sidebar.text_input("Benchmark display name", value="Benchmark",
                                       key="rs_const_name").strip() or "Benchmark"
    _use_screener = bool(_known) and st.sidebar.checkbox(
        f"Use the {len(_known)} screener tickers as constituents",
        value=False, key="rs_const_screener")
    if _use_screener:
        bench_members = _known
    else:
        _txt = st.sidebar.text_area("Constituents (comma or newline separated)",
                                    value="AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
                                    height=110, key="rs_const_txt")
        bench_members = [t.strip().upper()
                         for t in _txt.replace("\n", ",").split(",") if t.strip()]
    st.sidebar.caption(f"{len(bench_members or [])} constituents — "
                       "downloads one series per name, so large indexes are slow.")

else:  # Official turnover
    bench_name = st.sidebar.text_input("Benchmark display name", value="Benchmark",
                                       key="rs_off_name").strip() or "Benchmark"
    _upload = st.sidebar.file_uploader(
        "CSV with a date column and a turnover column", type=["csv"], key="rs_off_csv")
    if _upload is not None:
        try:
            bench_turnover_df = pd.read_csv(_upload)
        except Exception as _e:
            st.sidebar.error(f"Could not read CSV: {_e}")
    else:
        st.sidebar.caption("Upload the exchange's reported daily total turnover.")

st.sidebar.subheader("⏱️ Interval")
interval = st.sidebar.number_input("Periods to report (Average RS window)",
                                   min_value=2, max_value=500, value=14, step=1,
                                   key="rs_interval")
bar_interval = st.sidebar.selectbox("Bar frequency", ["1d", "1wk", "1mo"],
                                    index=0, key="rs_bar_interval")

st.sidebar.subheader("💰 Market Cap (optional)")
_use_cap = st.sidebar.checkbox("Normalise RS by market-cap weight", value=False,
                               key="rs_use_cap",
                               help="RS ÷ (stock cap / index cap). 1.0 = trades exactly "
                                    "in line with its index weight.")
stock_cap = bench_cap = None
if _use_cap:
    stock_cap = st.sidebar.number_input("Stock market cap", min_value=0.0,
                                        value=0.0, step=1e9, format="%.0f",
                                        key="rs_stock_cap")
    bench_cap = st.sidebar.number_input("Benchmark total market cap", min_value=0.0,
                                        value=0.0, step=1e9, format="%.0f",
                                        key="rs_bench_cap")
    if not stock_cap or not bench_cap:
        st.sidebar.warning("Both caps must be greater than zero to normalise.")
        stock_cap = bench_cap = None


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _fetch(tickers, start, end, bar):
    return rsm.fetch_ohlcv(tickers, start, end, bar)


# ── Run ───────────────────────────────────────────────────────────────────────
if not _run and 'rs_table' not in st.session_state:
    st.info("Set the ticker, benchmark and interval in the sidebar, then press "
            "**Calculate RS**.")
    st.stop()

if _run:
    try:
        if not ticker:
            st.error("Enter a ticker first.")
            st.stop()

        with st.spinner(f"Downloading {ticker}…"):
            stock_data = _fetch(ticker, start_date, end_date, bar_interval)
        if ticker not in stock_data:
            st.error(f"No data returned for {ticker} in that date range.")
            st.stop()

        if bench_mode == "ETF proxy":
            with st.spinner(f"Downloading {bench_proxy}…"):
                proxy_data = _fetch(bench_proxy, start_date, end_date, bar_interval)
            if bench_proxy not in proxy_data:
                st.error(f"No data returned for benchmark proxy {bench_proxy}.")
                st.stop()
            bench_value = rsm.benchmark_trading_value_from_proxy(proxy_data[bench_proxy])

        elif bench_mode == "Index constituents":
            if not bench_members:
                st.error("Add at least one constituent ticker.")
                st.stop()
            with st.spinner(f"Downloading {len(bench_members)} constituents…"):
                members = _fetch(list(bench_members), start_date, end_date, bar_interval)
            if not members:
                st.error("No constituent data returned.")
                st.stop()
            if len(members) < len(bench_members):
                st.warning(f"{len(bench_members) - len(members)} of "
                           f"{len(bench_members)} constituents returned no data and "
                           "were excluded from benchmark turnover.")
            bench_value = rsm.benchmark_trading_value_from_constituents(
                members, min_coverage=0.5)

        else:
            if bench_turnover_df is None:
                st.error("Upload a turnover CSV, or switch to another benchmark mode.")
                st.stop()
            bench_value = rsm.turnover_series_from_frame(bench_turnover_df)

        table, summary = rsm.build_rs_table(
            stock_data[ticker], bench_value,
            benchmark_name=bench_name,
            interval=int(interval),
            stock_market_cap=stock_cap,
            benchmark_market_cap=bench_cap,
        )

        if len(table) <= 1:
            st.error("No overlapping dates between the stock and the benchmark. "
                     "Check the date range and bar frequency.")
            st.stop()

        st.session_state['rs_table'] = table
        st.session_state['rs_summary'] = summary
        st.session_state['rs_ticker'] = ticker
        # Share the average with the other dashboards
        st.session_state.setdefault('rs_scores', {})[ticker] = summary['average_rs']

    except Exception as e:
        st.error(f"RS calculation failed: {e}")
        st.stop()

table = st.session_state['rs_table']
summary = st.session_state['rs_summary']
ticker = st.session_state['rs_ticker']
bench_col = next(c for c in table.columns if 'Trading Value' in c and c != 'Stock Trading Value')

# ── Metrics ───────────────────────────────────────────────────────────────────
_rows = table.iloc[:-1]           # drop the Average RS summary row
_avg_row = table.iloc[-1]

_m1, _m2, _m3, _m4 = st.columns(4)
_m1.metric(f"Average RS ({summary['interval']} periods)", f"{summary['average_rs']:.6f}")
_m2.metric("Latest RS", f"{summary['last']:.6f}",
           delta=f"{(summary['last'] - summary['average_rs']):+.6f} vs avg")
_arrow = {"Up": "📈", "Down": "📉", "Flat": "➡️"}.get(summary['direction'], "❔")
_m3.metric("Trend", f"{_arrow} {summary['direction']}",
           delta=f"{summary['slope_pct_per_period']:+.2f}% / period")
if summary.get('average_rs_cap_adjusted') is not None:
    _m4.metric("Avg RS, cap-adjusted", f"{summary['average_rs_cap_adjusted']:.3f}",
               help="1.0 = trades exactly in line with its index weight")
else:
    _m4.metric("Mean stock turnover", f"{_rows['Stock Trading Value'].mean():,.0f}")

st.caption(f"**{ticker}** vs **{summary['benchmark']}** — "
           f"RS moved {summary['first']:.6f} → {summary['last']:.6f} over "
           f"{summary['periods']} periods.")

if bench_mode == "ETF proxy":
    st.caption("⚠️ With an ETF proxy the RS *level* is a ratio to the ETF's own turnover, "
               "not the stock's share of index turnover. The trend and relative "
               "comparisons remain valid.")

# ── Chart ─────────────────────────────────────────────────────────────────────
_fig = go.Figure()
_fig.add_trace(go.Scatter(x=_rows['Date'], y=_rows['RS'], mode='lines+markers',
                          name='RS', line=dict(width=2)))
_fig.add_trace(go.Scatter(x=_rows['Date'],
                          y=rsm.rolling_average_rs(_rows['RS'], summary['interval']),
                          mode='lines', name=f"Rolling avg ({summary['interval']})",
                          line=dict(width=2, dash='dot')))
_fig.add_hline(y=summary['average_rs'], line_dash="dash", line_color="grey",
               annotation_text=f"Average RS {summary['average_rs']:.6f}",
               annotation_position="top left")
_fig.update_layout(title=f"{ticker} Relative Strength vs {summary['benchmark']}",
                   xaxis_title="Date", yaxis_title="RS (turnover share)",
                   hovermode="x unified", height=430,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02,
                               xanchor="right", x=1))
st.plotly_chart(_fig, use_container_width=True)

# ── Turnover comparison ───────────────────────────────────────────────────────
with st.expander("📊 Underlying trading values"):
    _fig2 = go.Figure()
    _fig2.add_trace(go.Bar(x=_rows['Date'], y=_rows['Stock Trading Value'],
                           name=f"{ticker} turnover"))
    _fig2.add_trace(go.Scatter(x=_rows['Date'], y=_rows[bench_col], mode='lines',
                               name=f"{summary['benchmark']} turnover", yaxis='y2'))
    _fig2.update_layout(height=360, hovermode="x unified",
                        yaxis=dict(title=f"{ticker}"),
                        yaxis2=dict(title=summary['benchmark'], overlaying='y',
                                    side='right', showgrid=False),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="right", x=1))
    st.plotly_chart(_fig2, use_container_width=True)

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader("📋 RS Table")
_display = rsm.format_rs_table(table).copy()
_display['Date'] = _display['Date'].map(
    lambda d: d.strftime('%Y-%m-%d') if isinstance(d, (pd.Timestamp, _dt.date)) else str(d))
st.dataframe(_display, use_container_width=True, hide_index=True,
             height=min(600, 38 * (len(_display) + 1)))

st.download_button(
    "⬇️ Download RS table (CSV)",
    data=table.to_csv(index=False).encode('utf-8'),
    file_name=f"RS_{ticker}_{summary['benchmark'].replace(' ', '_')}_{summary['interval']}p.csv",
    mime="text/csv",
)
