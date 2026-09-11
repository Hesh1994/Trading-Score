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

try:
    from canslim_module import (fetch_fmp_exchange_tickers, fetch_fmp_available_exchanges,
                                build_country_exchange_map, fetch_ticker_sectors,
                                COUNTRY_EXCHANGES)
    _canslim_ok = True
except ImportError:
    _canslim_ok = False

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

# Pull tickers and exchange selection already loaded by the scoring
# dashboard, if any.
_known = list(dict.fromkeys(
    list(st.session_state.get('ta_ticker_list', [])) +
    list(st.session_state.get('canslim_ticker_list', []))
))
_fmp_key = st.session_state.get('fmp_key_value', '')

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

# ── Benchmark — Ticker Finder only ──────────────────────────────────────────
# Exact replica of the Scoring Dashboard's "Ticker Finder": Country →
# Exchange → Load Tickers → Load Sectors → Sector filter → ticker
# multiselect with Add Selected / Add All. Builds the benchmark's
# constituent list directly, sharing the same FMP ticker/sector caches as
# the Scoring Dashboard so nothing is fetched twice.
st.sidebar.subheader("🏛️ Benchmark — Ticker Finder")

bench_name = st.sidebar.text_input("Benchmark display name", value="Custom Benchmark",
                                   key="rs_finder_bench_name").strip() or "Custom Benchmark"

if 'rs_bench_ticker_list' not in st.session_state:
    st.session_state['rs_bench_ticker_list'] = []

if not _canslim_ok:
    st.sidebar.warning("canslim_module not found — Ticker Finder unavailable.")
elif not _fmp_key:
    st.sidebar.caption("⬆️ Enter an FMP API key (on the Scoring Dashboard) to use the Ticker Finder.")
else:
    @st.cache_data(ttl=3600, show_spinner=False)
    def _rs_load_exchange_map(api_key):
        raw = fetch_fmp_available_exchanges(api_key)
        if raw:
            return build_country_exchange_map(raw)
        return COUNTRY_EXCHANGES

    _fc_map = _rs_load_exchange_map(_fmp_key)

    _fc_countries = st.sidebar.multiselect(
        "Country", options=sorted(_fc_map.keys()), key="rs_finder_country",
        placeholder="Search and select countries…",
    )

    _fc_exc_codes = []
    _fc_exc_label = None
    if _fc_countries:
        _fc_pairs = []
        for _c in _fc_countries:
            _fc_pairs.extend(_fc_map[_c])
        _fc_label_to_code = {lbl: code for code, lbl in _fc_pairs}
        _fc_sel_labels = st.sidebar.multiselect(
            "Exchange", options=list(_fc_label_to_code.keys()), key="rs_finder_exchange",
            placeholder="All exchanges (leave empty for all)…",
        )
        if _fc_sel_labels:
            _fc_exc_codes = [_fc_label_to_code[l] for l in _fc_sel_labels]
            _fc_exc_label = ", ".join(_fc_sel_labels)
        else:
            _fc_exc_codes = [code for code, _ in _fc_pairs]
            _fc_exc_label = f"All exchanges ({len(_fc_exc_codes)} selected)"

    st.sidebar.markdown("**📋 Load Exchange Tickers**")
    if not _fc_exc_codes:
        st.sidebar.caption("⬆️ Select a country above first.")
    else:
        _fc_ck  = f"ta_tickers_{'_'.join(sorted(_fc_exc_codes))}"
        _fc_sck = f"ta_sectors_{'_'.join(sorted(_fc_exc_codes))}"
        _fc_loaded  = bool(st.session_state.get(_fc_ck))
        _fc_sloaded = bool(st.session_state.get(_fc_sck))

        _flc, _frc = st.sidebar.columns(2)
        if _flc.button("📋 Load Tickers", key="rs_finder_load_btn",
                       use_container_width=True, disabled=_fc_loaded):
            with st.spinner(f"Loading tickers for {_fc_exc_label}…"):
                try:
                    if _fc_exc_codes == ["__ALL__"]:
                        _fc_tickers = fetch_fmp_exchange_tickers("__ALL__", _fmp_key)
                    else:
                        _fc_combined = {}
                        for _c in _fc_exc_codes:
                            for _s, _n in fetch_fmp_exchange_tickers(_c, _fmp_key):
                                _fc_combined[_s] = _n
                        _fc_tickers = sorted(_fc_combined.items(), key=lambda x: x[0])
                    st.session_state[_fc_ck] = _fc_tickers
                    st.rerun()
                except RuntimeError as _e:
                    st.sidebar.error(str(_e))
        if _frc.button("🔄 Reload", key="rs_finder_reload_btn",
                       use_container_width=True, disabled=not _fc_loaded):
            st.session_state.pop(_fc_ck, None)
            st.session_state.pop(_fc_sck, None)
            st.rerun()

        if _fc_loaded:
            _fc_tickers = st.session_state[_fc_ck]
            st.sidebar.caption(f"{len(_fc_tickers):,} tickers loaded from FMP")

            if not _fc_sloaded:
                if st.sidebar.button("🏭 Load Sectors", key="rs_finder_load_sectors_btn",
                                      use_container_width=True):
                    _fc_syms = [s for s, _ in _fc_tickers]
                    _fc_pb = st.sidebar.progress(0, text="Fetching sectors…")
                    _fc_pt = st.sidebar.empty()
                    def _fc_pcb(done, total):
                        _fc_pb.progress(done / total, text=f"Sectors: {done}/{total}")
                        _fc_pt.caption(f"{done}/{total} processed")
                    _fc_smap = fetch_ticker_sectors(_fc_syms, _fmp_key, progress_cb=_fc_pcb)
                    _fc_pb.empty(); _fc_pt.empty()
                    st.session_state[_fc_sck] = _fc_smap
                    st.rerun()
            else:
                _fc_smap = st.session_state.get(_fc_sck, {})
                _fc_filled = sum(1 for v in _fc_smap.values() if v)
                st.sidebar.caption(f"Sectors: {_fc_filled:,} / {len(_fc_smap):,} classified")

            _fc_smap = st.session_state.get(_fc_sck, {})
            _fc_avail_sectors = sorted({v for v in _fc_smap.values() if v})
            _fc_sec_choice = st.sidebar.multiselect(
                "🏭 Filter by Sector", options=_fc_avail_sectors,
                key="rs_finder_sector_filter", placeholder="All sectors (no filter)",
            )
            _fc_in_sec = (
                _fc_tickers if (not _fc_sec_choice or not _fc_smap)
                else [(s, n) for s, n in _fc_tickers if _fc_smap.get(s, "") in set(_fc_sec_choice)]
            )

            _fc_opts = [f"{s}  —  {n}" for s, n in _fc_in_sec[:1000]]
            _fc_chosen = st.sidebar.multiselect(
                f"Select ticker ({len(_fc_in_sec):,} available)", options=_fc_opts,
                key="rs_finder_ticker_select", placeholder="Search and select tickers…",
            )
            _fb1, _fb2 = st.sidebar.columns(2)
            if _fb1.button("➕ Add Selected", key="rs_finder_add_btn",
                          use_container_width=True, disabled=not _fc_chosen):
                _fc_before = len(st.session_state['rs_bench_ticker_list'])
                for _lbl in _fc_chosen:
                    _sym = _lbl.split("  —  ")[0].strip()
                    if _sym not in st.session_state['rs_bench_ticker_list']:
                        st.session_state['rs_bench_ticker_list'].append(_sym)
                _fc_added = len(st.session_state['rs_bench_ticker_list']) - _fc_before
                st.session_state.pop("rs_finder_ticker_select", None)
                st.sidebar.success(f"Added {_fc_added} ticker(s)")
                st.rerun()
            if _fb2.button("➕ Add All", key="rs_finder_add_all_btn",
                          use_container_width=True):
                _fc_before = len(st.session_state['rs_bench_ticker_list'])
                for _s, _ in _fc_in_sec:
                    if _s not in st.session_state['rs_bench_ticker_list']:
                        st.session_state['rs_bench_ticker_list'].append(_s)
                _fc_added = len(st.session_state['rs_bench_ticker_list']) - _fc_before
                st.sidebar.success(f"Added {_fc_added} ticker(s)")
                st.rerun()
        else:
            st.sidebar.caption("Click **Load Tickers** to browse listed stocks.")

bench_members = list(st.session_state['rs_bench_ticker_list'])
if bench_members:
    st.sidebar.caption(f"**{len(bench_members)} ticker(s) in benchmark**")
    if st.sidebar.button("🗑️ Clear benchmark list", key="rs_finder_clear_btn"):
        st.session_state['rs_bench_ticker_list'] = []
        st.rerun()

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

        if not bench_members:
            st.error("The benchmark has no tickers yet — use the Ticker Finder in the "
                     "sidebar to add some (Load Tickers, then Add Selected / Add All).")
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
        # Share the averages with the other dashboards
        st.session_state.setdefault('rs_scores', {}).update(
            {t: s['average_rs'] for t, s in rs_summaries.items()})

    except Exception as e:
        st.error(f"RS calculation failed: {e}")
        st.stop()

rs_tables = st.session_state['rs_tables']
rs_summaries = st.session_state['rs_summaries']
rs_tickers = st.session_state['rs_tickers']

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

where **Trading Value = Close × Volume** for each bar. This is a true sum of
the benchmark's own constituent stocks' turnover — see "Benchmark: the
Ticker Finder" below.

The idea: a rising RS means the stock is capturing a growing *share of the
market's money flow* relative to the benchmark, regardless of whether its
price is up or down that day. It's a participation/liquidity signal, distinct
from a price-momentum signal.

### Step by step

1. **Stock(s)** — pick one or more tickers, imported straight from the
   Scoring Dashboard's ticker list (or type your own). Every ticker you pick
   is scored against the **same benchmark**, so they're directly comparable.
2. **Trading value** is computed per bar for each stock: `Close × Volume`.
3. **Benchmark trading value** is computed the same way, summed across every
   ticker you've added to the benchmark via the Ticker Finder (below).
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

### Benchmark: the Ticker Finder

There's a single way to build the benchmark, and it's always visible in the
sidebar right away — no mode picker, no ETF shortcuts. It's the *exact same*
tool as the Scoring Dashboard's Ticker Finder:

1. **Country** — pick one or more countries.
2. **Exchange** — optionally narrow to specific exchanges (leave empty for
   every exchange in the chosen countries).
3. **Load Tickers** — pulls the listed tickers for that selection via FMP.
4. **Load Sectors** *(optional)* — classifies each ticker's sector, so you
   can then **Filter by Sector**.
5. **Select ticker** — search and multiselect from the (optionally
   sector-filtered) list, then **➕ Add Selected** or **➕ Add All** to add
   them to the benchmark's constituent list, shown as a running count in
   the sidebar with a **🗑️ Clear benchmark list** button to start over.

This shares the same FMP ticker/sector caches as the Scoring Dashboard, so
anything already loaded there is reused instead of re-fetched. The result is
always a **true constituent sum**: `Close × Volume` added up across every
ticker you've added, whether that's a handful of hand-picked peers, an
entire sector, or an entire exchange — never an ETF proxy standing in for
it. Requires an FMP API key, entered on the Scoring Dashboard.

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
  computed with the *same benchmark*; an RS of 0.05 against a 20-stock
  benchmark isn't comparable to an RS of 0.05 against a 200-stock one.

### Caveats

- Turnover here is approximated as `Close × Volume`, not any exchange's
  officially reported value-traded figure.
- Corporate actions (splits, big single-day volume spikes from index
  rebalances, etc.) can distort both the stock's and the benchmark's trading
  value for that bar.
- With very few overlapping dates (e.g. mismatched bar frequency or date
  range between the stock and benchmark), the RS series and trend become
  unreliable — each ticker needs at least one overlapping period beyond the
  summary row to be included at all.
""")
