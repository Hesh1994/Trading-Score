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

# Pull tickers, exchange selection, and sector map already loaded by the
# scoring dashboard, if any.
_known = list(dict.fromkeys(
    list(st.session_state.get('ta_ticker_list', [])) +
    list(st.session_state.get('canslim_ticker_list', []))
))
_exch_codes  = list(st.session_state.get('ta_exchange_codes', []))
_exch_label  = st.session_state.get('ta_exchange_label')
_exch_all    = list(st.session_state.get('ta_exchange_all_tickers', []))   # [(sym, name), ...]
_sector_map  = dict(st.session_state.get('ta_exchange_sector_map', {}))
_avail_sectors = sorted({v for v in _sector_map.values() if v})

# ── Stocks to analyse ───────────────────────────────────────────────────────
st.sidebar.subheader("🎯 Stock(s)")
if _known:
    _src = st.sidebar.radio("Ticker source", ["From screener", "Type manually"],
                            horizontal=True, key="rs_src")
else:
    _src = "Type manually"

if _src == "From screener":
    tickers = st.sidebar.multiselect(
        "Tickers (imported from the Scoring Dashboard)",
        options=_known, default=_known, key="rs_ticker_ms",
    )
else:
    _txt = st.sidebar.text_input("Ticker(s) — comma separated", value="AAPL",
                                 key="rs_ticker_txt")
    tickers = [t.strip().upper() for t in _txt.split(",") if t.strip()]

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

# ── Benchmark — single selector ─────────────────────────────────────────────
st.sidebar.subheader("🏛️ Benchmark")

_auto_proxy = rsm.exchange_index_proxy(_exch_codes)

_bench_options = []  # list of (key, label)
if _auto_proxy:
    _auto_etf, _auto_name = _auto_proxy
    _bench_options.append(
        ("auto_index", f"🏛️ {_auto_name} — auto index for {_exch_label} ({_auto_etf})"))

_bench_options += [
    ("spy", "📈 S&P 500 (SPY)"),
    ("qqq", "📈 Nasdaq 100 (QQQ)"),
    ("dia", "📈 Dow 30 (DIA)"),
    ("iwm", "📈 Russell 2000 (IWM)"),
    ("ksa", "📈 Saudi TASI (KSA)"),
]
if _known:
    _bench_options.append(("group", f"👥 My screener tickers ({len(_known)})"))
if _avail_sectors:
    _bench_options.append(("sector", "🏭 Sector peers"))
if _exch_all:
    _bench_options.append(
        ("market", f"🌍 Entire {_exch_label or 'exchange'} turnover ({len(_exch_all):,} tickers)"))
_bench_options.append(("custom_list", "✍️ Custom ticker list"))
_bench_options.append(("custom_etf", "🔤 Custom ETF ticker"))
_bench_options.append(("upload", "📄 Official turnover (upload CSV)"))

_bench_keys   = [k for k, _ in _bench_options]
_bench_labels = {k: l for k, l in _bench_options}
bench_key = st.sidebar.selectbox(
    "Benchmark trading value from",
    options=_bench_keys,
    format_func=lambda k: _bench_labels[k],
    key="rs_bench_choice",
    help="One benchmark drives the whole page — pick where its turnover comes from.",
)

_PROXY_NAMES = {
    "spy": ("SPY", "S&P 500"), "qqq": ("QQQ", "Nasdaq 100"),
    "dia": ("DIA", "Dow 30"), "iwm": ("IWM", "Russell 2000"),
    "ksa": ("KSA", "Saudi TASI"),
}

bench_proxy = bench_members = bench_turnover_df = None
bench_name = "Benchmark"
is_proxy_mode = False

if bench_key == "auto_index":
    bench_proxy, bench_name = _auto_etf, _auto_name
    is_proxy_mode = True

elif bench_key in _PROXY_NAMES:
    bench_proxy, bench_name = _PROXY_NAMES[bench_key]
    is_proxy_mode = True

elif bench_key == "custom_etf":
    bench_proxy = st.sidebar.text_input("Proxy ETF ticker", value="SPY",
                                        key="rs_proxy_custom").strip().upper()
    bench_name = st.sidebar.text_input("Benchmark display name", value=bench_proxy,
                                       key="rs_proxy_name").strip() or bench_proxy
    is_proxy_mode = True

elif bench_key == "group":
    bench_name = "My Screener Tickers"
    bench_members = [t for t in _known if t not in tickers] or _known
    st.sidebar.caption(f"{len(bench_members)} constituents from the Scoring Dashboard list — "
                       "downloads one series per name.")

elif bench_key == "sector":
    _sector_choice = st.sidebar.selectbox("Sector", _avail_sectors, key="rs_sector_choice")
    bench_name = f"{_sector_choice} Sector"
    bench_members = [s for s, sec in _sector_map.items() if sec == _sector_choice]
    st.sidebar.caption(f"{len(bench_members)} constituents in **{_sector_choice}** "
                       f"(from {_exch_label or 'the loaded exchange'}).")

elif bench_key == "market":
    bench_name = f"{_exch_label or 'Exchange'} Market"
    _max_n = st.sidebar.number_input(
        "Max constituents (speed vs accuracy)", min_value=10,
        max_value=max(10, len(_exch_all)), value=min(300, len(_exch_all)), step=10,
        key="rs_market_cap_n",
        help="Summing Close×Volume across every listed ticker gives the truest "
             "market-turnover figure, but downloads one series per name.")
    bench_members = [s for s, _ in _exch_all[:int(_max_n)]]
    st.sidebar.caption(f"Using {len(bench_members)} of {len(_exch_all):,} listed tickers.")

elif bench_key == "custom_list":
    bench_name = st.sidebar.text_input("Benchmark display name", value="Benchmark",
                                       key="rs_const_name").strip() or "Benchmark"
    _txt2 = st.sidebar.text_area("Constituents (comma or newline separated)",
                                 value="AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
                                 height=110, key="rs_const_txt")
    bench_members = [t.strip().upper()
                     for t in _txt2.replace("\n", ",").split(",") if t.strip()]
    st.sidebar.caption(f"{len(bench_members)} constituents — "
                       "downloads one series per name, so large lists are slow.")

else:  # upload
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
                                    "in line with its index weight. Applies to a single "
                                    "drill-down ticker at a time when analysing several.")
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
if not _run and 'rs_tables' not in st.session_state:
    st.info("Set the ticker(s), benchmark and interval in the sidebar, then press "
            "**Calculate RS**.")
    st.stop()

if _run:
    try:
        if not tickers:
            st.error("Select or enter at least one ticker first.")
            st.stop()

        with st.spinner(f"Downloading {len(tickers)} ticker(s)…"):
            stock_data = _fetch(list(tickers), start_date, end_date, bar_interval)
        if not stock_data:
            st.error("No data returned for the selected ticker(s) in that date range.")
            st.stop()
        _missing = [t for t in tickers if t not in stock_data]
        if _missing:
            st.warning(f"No data for: {', '.join(_missing)} — excluded.")

        if bench_proxy is not None:
            with st.spinner(f"Downloading benchmark {bench_proxy}…"):
                proxy_data = _fetch(bench_proxy, start_date, end_date, bar_interval)
            if bench_proxy not in proxy_data:
                st.error(f"No data returned for benchmark proxy {bench_proxy}.")
                st.stop()
            bench_value = rsm.benchmark_trading_value_from_proxy(proxy_data[bench_proxy])

        elif bench_members is not None:
            if not bench_members:
                st.error("The benchmark has no constituents — check the sidebar selection.")
                st.stop()
            with st.spinner(f"Downloading {len(bench_members)} benchmark constituents…"):
                members = _fetch(list(bench_members), start_date, end_date, bar_interval)
            if not members:
                st.error("No constituent data returned for the benchmark.")
                st.stop()
            if len(members) < len(bench_members):
                st.warning(f"{len(bench_members) - len(members)} of "
                           f"{len(bench_members)} benchmark constituents returned no data "
                           "and were excluded from benchmark turnover.")
            bench_value = rsm.benchmark_trading_value_from_constituents(
                members, min_coverage=0.5)

        else:
            if bench_turnover_df is None:
                st.error("Upload a turnover CSV, or switch to another benchmark mode.")
                st.stop()
            bench_value = rsm.turnover_series_from_frame(bench_turnover_df)

        rs_tables, rs_summaries = {}, {}
        for _t in tickers:
            if _t not in stock_data:
                continue
            _table, _summary = rsm.build_rs_table(
                stock_data[_t], bench_value,
                benchmark_name=bench_name,
                interval=int(interval),
                stock_market_cap=stock_cap,
                benchmark_market_cap=bench_cap,
            )
            if len(_table) > 1:
                rs_tables[_t] = _table
                rs_summaries[_t] = _summary

        if not rs_tables:
            st.error("No overlapping dates between the stock(s) and the benchmark. "
                     "Check the date range and bar frequency.")
            st.stop()

        st.session_state['rs_tables'] = rs_tables
        st.session_state['rs_summaries'] = rs_summaries
        st.session_state['rs_tickers'] = list(rs_tables.keys())
        st.session_state['rs_is_proxy_mode'] = is_proxy_mode
        # Share the averages with the other dashboards
        st.session_state.setdefault('rs_scores', {}).update(
            {t: s['average_rs'] for t, s in rs_summaries.items()})

    except Exception as e:
        st.error(f"RS calculation failed: {e}")
        st.stop()

rs_tables = st.session_state['rs_tables']
rs_summaries = st.session_state['rs_summaries']
rs_tickers = st.session_state['rs_tickers']
_is_proxy_mode = st.session_state.get('rs_is_proxy_mode', False)

tab_results, tab_notes = st.tabs(["📊 Results", "📖 Notes: How RS Works"])

with tab_results:
    _bench_display = next(iter(rs_summaries.values()))['benchmark']

    # ── Multi-ticker summary ────────────────────────────────────────────────
    if len(rs_tickers) > 1:
        st.subheader(f"📋 RS Summary — {len(rs_tickers)} tickers vs {_bench_display}")
        _summary_rows = []
        for _t in rs_tickers:
            _s = rs_summaries[_t]
            _arrow = {"Up": "📈", "Down": "📉", "Flat": "➡️"}.get(_s['direction'], "❔")
            _summary_rows.append({
                'Ticker': _t,
                'Average RS': _s['average_rs'],
                'Latest RS': _s['last'],
                'Trend': f"{_arrow} {_s['direction']}",
                '% / period': _s['slope_pct_per_period'],
            })
        _summary_df = pd.DataFrame(_summary_rows).sort_values('Average RS', ascending=False)
        st.dataframe(
            _summary_df, use_container_width=True, hide_index=True,
            column_config={
                'Average RS': st.column_config.NumberColumn('Average RS', format="%.6f"),
                'Latest RS':  st.column_config.NumberColumn('Latest RS', format="%.6f"),
                '% / period': st.column_config.NumberColumn('% / period', format="%.2f%%"),
            },
        )
        st.download_button(
            "⬇️ Download RS summary (CSV)",
            data=_summary_df.to_csv(index=False).encode('utf-8'),
            file_name=f"RS_summary_{_bench_display.replace(' ', '_')}_{int(interval)}p.csv",
            mime="text/csv",
        )
        st.markdown("---")
        st.subheader("🔎 Drill-down")

    ticker = st.selectbox("Ticker", rs_tickers, key="rs_drill_ticker") \
        if len(rs_tickers) > 1 else rs_tickers[0]

    table = rs_tables[ticker]
    summary = rs_summaries[ticker]
    bench_col = next(c for c in table.columns if 'Trading Value' in c and c != 'Stock Trading Value')

    # ── Metrics ───────────────────────────────────────────────────────────
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

    if _is_proxy_mode:
        st.caption("⚠️ With an ETF/index proxy the RS *level* is a ratio to the proxy's own "
                   "turnover, not the stock's share of index turnover. The trend and "
                   "relative comparisons remain valid.")

    # ── Chart ─────────────────────────────────────────────────────────────
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

    # ── Turnover comparison ──────────────────────────────────────────────
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

    # ── Table ─────────────────────────────────────────────────────────────
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

# ── Notes tab ────────────────────────────────────────────────────────────────
with tab_notes:
    st.markdown(r"""
### What this page measures

This is **turnover-based Relative Strength (RS)** — not the classic "price ÷
price" relative-strength line, and not the RSI oscillator. Instead of
comparing how far a stock's *price* has moved versus a benchmark, it compares
how much *money is trading hands* in the stock versus the benchmark:

$$
\text{RS}(t) = \frac{\text{Stock Trading Value}(t)}{\text{Benchmark Trading Value}(t)}
$$

where **Trading Value = Close × Volume** for each bar. This is a proxy for
daily turnover (the exchange's own reported turnover, if you have it, is more
accurate — see "Benchmark options" below).

The idea: a rising RS means the stock is capturing a growing *share of the
market's money flow* relative to the benchmark, regardless of whether its
price is up or down that day. It's a participation/liquidity signal, distinct
from a price-momentum signal.

### Step by step

1. **Stock(s)** — pick one or more tickers, imported straight from the
   Scoring Dashboard's ticker list (or type your own). Every ticker you pick
   is scored against the **same benchmark**, so they're directly comparable.
2. **Trading value** is computed per bar for each stock: `Close × Volume`.
3. **Benchmark trading value** is computed the same way, using whichever
   single benchmark option you picked in the sidebar (below).
4. **RS** is the ratio of the two, aligned on the dates both series share.
   A benchmark value of zero (or a missing overlapping date) becomes blank
   rather than an artificial spike.
5. **Average RS** is the mean of RS over the last *N* periods you set as the
   "Interval" — this is the headline number for each ticker, and with
   multiple tickers selected it drives the ranked summary table at the top.
6. **Trend** fits a straight line (ordinary least squares) through the RS
   values in that window. The slope is expressed as a **% of the window's
   mean RS per period**:
   - **Up** — slope is at least +0.5% of mean RS per period
   - **Down** — slope is at most −0.5% of mean RS per period
   - **Flat** — anything in between

### Benchmark — one selector, several sources

The **Benchmark** dropdown in the sidebar is the single control for where
turnover comes from. Depending on what's selected it reveals the matching
follow-up input (a sector picker, a ticker box, a file uploader):

- **🏛️ Auto index for your exchange** *(when you've picked a country/exchange
  on the Scoring Dashboard)* — a liquid ETF that tracks that exchange's
  market, chosen automatically (e.g. Tadawul → KSA, LSE → EWU, NASDAQ → QQQ).
- **📈 Major indices** — S&P 500 (SPY), Nasdaq 100 (QQQ), Dow 30 (DIA),
  Russell 2000 (IWM), Saudi TASI (KSA) — fixed presets regardless of exchange.
- **👥 My screener tickers** — the exact group of tickers you've built up on
  the Scoring Dashboard / CANSLIM pages, summed into one turnover series.
  Good for "how is this stock doing versus my own watchlist as a whole."
- **🏭 Sector peers** — every ticker sharing the chosen sector (from the
  sector classification loaded on the Scoring Dashboard), summed together.
- **🌍 Entire exchange turnover** — sums `Close × Volume` across every ticker
  loaded for your chosen exchange, i.e. the whole market's turnover. A "Max
  constituents" control caps how many names are downloaded, trading accuracy
  for speed.
- **✍️ Custom ticker list** — paste any comma/newline-separated list of
  symbols to build a bespoke peer group.
- **🔤 Custom ETF ticker** — any single ETF or index ticker as a proxy.
- **📄 Official turnover (upload CSV)** — if the exchange publishes its own
  daily total turnover figure, upload it directly. This is the most accurate
  option since it isn't a proxy or an approximation from constituent data.

An **ETF/index proxy** (auto index, majors, or custom ETF) is fastest, but
the RS *level* is only a ratio to that ETF's own turnover — it is **not**
the stock's true share of total market turnover, since an ETF trades a tiny
fraction of its underlying index's volume. The **trend** (rising/falling)
stays meaningful either way, and the app flags this with a caption whenever
a proxy is in use. The **constituent-sum** options (screener group, sector,
entire exchange, custom list) give a true "share of turnover" reading
instead, at the cost of one download per constituent — a day is blanked out
if fewer than 50% of constituents reported data, so an outage can't quietly
deflate the benchmark.

### Reading the numbers

- **RS Summary table** *(with multiple tickers)* — every selected ticker
  ranked by Average RS against the one benchmark, so you can see at a glance
  which stocks are capturing the most (or least) relative turnover, and
  whether each is trending up or down.
- **Average RS** — the headline metric per ticker; only compare it across
  tickers computed with the *same benchmark*.
- **Latest RS vs average** — is the most recent reading above or below its
  own recent norm?
- **Trend arrow** — 📈 Up / 📉 Down / ➡️ Flat, from the slope test above.
- **RS, cap-adjusted** *(optional, drill-down ticker only)* — enter the
  stock's market cap and the benchmark's total market cap to divide RS by
  the stock's index *weight* (`stock cap ÷ benchmark cap`). **1.0** means the
  stock trades exactly in proportion to its index weight; **above 1.0** means
  disproportionate interest; **below 1.0** means less.
- **RS is a ratio, not a percentage or a price** — only compare RS values
  computed with the *same benchmark*; an RS of 0.05 against an ETF proxy is
  not comparable to an RS of 0.05 against a full sector sum.

### Caveats

- Turnover here is approximated as `Close × Volume`, not the exchange's
  official value-traded figure, unless you use the "Official turnover" option.
- Corporate actions (splits, big single-day volume spikes from index
  rebalances, etc.) can distort both the stock's and the benchmark's trading
  value for that bar.
- With very few overlapping dates (e.g. mismatched bar frequency or date
  range between the stock and benchmark), the RS series and trend become
  unreliable — each ticker needs at least one overlapping period beyond the
  summary row to be included at all.
""")
