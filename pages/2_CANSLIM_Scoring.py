"""
CANSLIM Multi-Ticker Scoring & Ranking Page
Scores any number of companies on the CANSLIM fundamentals methodology
and presents a ranked table with per-ticker drill-down.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import requests as _req

try:
    from canslim_module import (score_canslim_universe, COUNTRY_EXCHANGES,
                                EXCHANGE_SUFFIX, validate_ticker_fmp, format_ticker,
                                fetch_fmp_exchange_tickers)
except ImportError as _e:
    st.error(f"Cannot import canslim_module: {_e}\nMake sure canslim_module.py is in the repo root.")
    st.stop()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="CANSLIM Scoring",
    page_icon="📈",
    layout="wide"
)

st.title("📈 CANSLIM Multi-Ticker Scoring & Ranking")
st.caption(
    "Scores each ticker across 10 CANSLIM criteria (10 pts each, max 100). "
    "Data via yfinance — needs at least 8 quarters / 6 years of history."
)

# ============================================================================
# HELPERS
# ============================================================================

def _pct(val, d=1):
    return f"{val * 100:.{d}f}%" if val is not None else "—"


def _traffic(met):
    return "✅" if met is True else ("❌" if met is False else "⚠️ no data")


def _fmt_criterion(label, val):
    if "Acceleration" in label:
        return "✅ TRUE" if val is True else ("❌ FALSE" if val is False else "no data")
    return _pct(val)


# ============================================================================
# SIDEBAR — DATA SOURCE (FMP API KEY)
# ============================================================================

st.sidebar.header("⚙️ CANSLIM Configuration")

st.sidebar.subheader("🔑 Data Source")
fmp_key = st.sidebar.text_input(
    "FMP API Key (optional)",
    type="password",
    placeholder="Leave blank to use yfinance",
    key="canslim_fmp_key",
    help="Get a free key at financialmodelingprep.com — provides cleaner fundamental data than yfinance."
)
if fmp_key:
    st.sidebar.success("FMP key set — using FMP data")
else:
    st.sidebar.caption("No key — using yfinance")

# ============================================================================
# SIDEBAR — TICKER FINDER
# ============================================================================

st.sidebar.subheader("🌍 Ticker Finder")

if 'canslim_ticker_list' not in st.session_state:
    st.session_state['canslim_ticker_list'] = []

# ── Country & Exchange ────────────────────────────────────────────────────
country_options = ["-- Select country --"] + sorted(COUNTRY_EXCHANGES.keys())
selected_country = st.sidebar.selectbox("Country", country_options, key="canslim_country")

selected_exchange_code = None
if selected_country != "-- Select country --":
    exchange_list = COUNTRY_EXCHANGES[selected_country]
    exchange_labels = [label for _, label in exchange_list]
    selected_exchange_label = st.sidebar.selectbox("Exchange", exchange_labels, key="canslim_exchange")
    selected_exchange_code = next(
        code for code, label in exchange_list if label == selected_exchange_label
    )
    suffix = EXCHANGE_SUFFIX.get(selected_exchange_code, "")
    st.sidebar.caption(f"Ticker suffix for this exchange: `{suffix if suffix else '(none)'}`")

# ── Dynamic ticker list from FMP ──────────────────────────────────────────
st.sidebar.markdown("**📋 Load Exchange Tickers**")

if not selected_exchange_code:
    st.sidebar.caption("⬆️ Select a country and exchange above first.")
elif not fmp_key:
    st.sidebar.caption("⬆️ Enter your FMP API key above to load the full ticker list.")
else:
    cache_key = f"fmp_tickers_{selected_exchange_code}"
    already_loaded = bool(cache_key in st.session_state and st.session_state[cache_key])

    load_col, reload_col = st.sidebar.columns(2)
    if load_col.button("📋 Load Tickers", key="load_tickers_btn", use_container_width=True,
                        disabled=already_loaded):
        with st.spinner(f"Loading tickers for {selected_exchange_label}…"):
            try:
                tickers = fetch_fmp_exchange_tickers(selected_exchange_code, fmp_key)
                st.session_state[cache_key] = tickers
                st.rerun()
            except RuntimeError as _err:
                st.sidebar.error(str(_err))

    if reload_col.button("🔄 Reload", key="reload_tickers_btn", use_container_width=True,
                          disabled=not already_loaded):
        st.session_state.pop(cache_key, None)
        st.rerun()

    if already_loaded:
        tickers = st.session_state[cache_key]
        st.sidebar.caption(f"{len(tickers):,} tickers loaded from FMP")

        search_q = st.sidebar.text_input(
            "🔍 Search ticker or name", placeholder="e.g. ABUK or Commercial",
            key="ticker_search"
        )
        q = search_q.strip().upper()
        filtered = (
            [(s, n) for s, n in tickers if q in s.upper() or q in (n or "").upper()]
            if q else tickers
        )

        if filtered:
            options = [f"{s}  —  {n}" for s, n in filtered[:200]]
            chosen = st.sidebar.selectbox(
                f"Select ticker ({len(filtered):,} match{'es' if len(filtered) != 1 else ''})",
                ["-- select --"] + options,
                key="ticker_select"
            )
            if chosen != "-- select --":
                chosen_sym = chosen.split("  —  ")[0].strip()
                if st.sidebar.button("➕ Add selected", key="add_from_list_btn",
                                      use_container_width=True):
                    if chosen_sym not in st.session_state['canslim_ticker_list']:
                        st.session_state['canslim_ticker_list'].append(chosen_sym)
                        st.sidebar.success(f"Added **{chosen_sym}**")
                    else:
                        st.sidebar.info(f"**{chosen_sym}** already in list")
        else:
            st.sidebar.info("No tickers match your search.")
    else:
        st.sidebar.caption("Click **Load Tickers** to browse all listed stocks.")

# ── Manual entry ──────────────────────────────────────────────────────────
st.sidebar.markdown("**Or enter ticker manually:**")
raw_ticker = st.sidebar.text_input(
    "Ticker symbol", placeholder="e.g. ABUK or AAPL", key="canslim_raw_ticker"
)
if raw_ticker:
    formatted = format_ticker(raw_ticker, selected_exchange_code or "")
    col_v, col_a = st.sidebar.columns(2)
    if col_v.button("🔍 Verify", key="canslim_verify_btn", use_container_width=True):
        if fmp_key:
            name, exch, cur = validate_ticker_fmp(formatted, fmp_key)
            if name:
                st.sidebar.success(f"✅ **{formatted}** — {name} ({exch}, {cur})")
            else:
                st.sidebar.warning(
                    f"⚠️ **{formatted}** not found via FMP. "
                    "It may still be valid — use ➕ Add anyway."
                )
        else:
            st.sidebar.warning("Enter an FMP API key to verify tickers.")
    if col_a.button("➕ Add", key="canslim_add_btn", use_container_width=True):
        if formatted not in st.session_state['canslim_ticker_list']:
            st.session_state['canslim_ticker_list'].append(formatted)
            st.sidebar.success(f"Added **{formatted}**")
        else:
            st.sidebar.info(f"**{formatted}** already in list")

# ── Current ticker list ───────────────────────────────────────────────────
if st.session_state['canslim_ticker_list']:
    st.sidebar.markdown("**Tickers to analyse:**  " +
                        " · ".join(f"`{t}`" for t in st.session_state['canslim_ticker_list']))
    if st.sidebar.button("🗑️ Clear list", key="canslim_clear_list"):
        st.session_state['canslim_ticker_list'] = []

st.sidebar.divider()

# ============================================================================
# SIDEBAR — MANUAL TICKER ENTRY
# ============================================================================

st.sidebar.subheader("🎯 Tickers")
_default = (", ".join(st.session_state['canslim_ticker_list'])
            if st.session_state['canslim_ticker_list']
            else "AAPL, MSFT, GOOGL, NVDA, AMZN")
ticker_input = st.sidebar.text_area(
    "Or enter tickers manually (comma-separated)",
    value=_default,
    height=80,
    key="canslim_tickers",
)

st.sidebar.markdown(
    """
    **Scoring criteria (10 pts each)**
    - Recent Qtr EPS YoY > 20%
    - EPS Acceleration (3 Qtrs)
    - EPS 3Y Avg Growth > 25%
    - EPS 5Y Avg Growth > 25%
    - Recent Qtr Revenue YoY > 25%
    - Avg Revenue Growth (3Q) > 25%
    - Pretax Income Annual > 15%
    - Net Income Annual > 25%
    - ROE YoY Growth > 17%
    - Institutional Ownership > 35%
    """
)

# ============================================================================
# SIDEBAR — TEST FMP CONNECTION
# ============================================================================

if fmp_key:
    test_ticker = st.sidebar.text_input("Test ticker", value="AAPL", key="canslim_test_ticker")
    if st.sidebar.button("🔌 Test FMP Connection", key="canslim_test_btn"):
        _url = (f"https://financialmodelingprep.com/stable/income-statement"
                f"?symbol={test_ticker.upper()}&period=quarterly&limit=2&apikey={fmp_key}")
        try:
            _r = _req.get(_url, timeout=10)
            _data = _r.json()
            if isinstance(_data, list) and _data:
                st.sidebar.success(f"✅ FMP OK — got {len(_data)} quarters for {test_ticker.upper()}")
                st.sidebar.json(_data[0])
            elif isinstance(_data, dict) and "Error Message" in _data:
                st.sidebar.error(f"FMP error: {_data['Error Message']}")
            else:
                st.sidebar.warning(f"Unexpected response: {_data}")
        except Exception as _e:
            st.sidebar.error(f"Connection failed: {_e}")

run_btn = st.sidebar.button("🚀 Run CANSLIM Analysis", type="primary", use_container_width=True)

# ============================================================================
# MAIN — RUN ANALYSIS
# ============================================================================

if run_btn:
    manual  = [s.strip().upper() for s in ticker_input.replace("\n", ",").split(",") if s.strip()]
    finder  = st.session_state.get('canslim_ticker_list', [])
    symbols = list(dict.fromkeys(finder + manual))   # deduplicate, preserve order

    if not symbols:
        st.warning("Enter at least one ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching fundamental data for {len(symbols)} ticker(s)…"):
        st.session_state['canslim_results'] = score_canslim_universe(symbols, fmp_api_key=fmp_key or None)
        st.session_state['canslim_source']  = "FMP API" if fmp_key else "yfinance"

if st.session_state.get('canslim_results'):
    results      = st.session_state['canslim_results']
    source_label = st.session_state.get('canslim_source', 'FMP API')

    # ── Data fetch diagnostics ────────────────────────────────────────────
    all_errors = [(r['symbol'], e) for r in results for e in r.get('errors', [])]
    if all_errors:
        with st.expander(f"⚠️ Data fetch warnings ({len(all_errors)} issues via {source_label}) — click to expand"):
            for sym_err, msg in all_errors:
                st.write(f"**{sym_err}**: {msg}")
    else:
        st.success(f"✅ All data fetched successfully via {source_label}")

    # ── Sector filter ─────────────────────────────────────────────────────
    all_sectors = sorted({r['sector'] for r in results if r.get('sector')})
    sector_options = ["ALL"] + all_sectors
    selected_sector = st.selectbox(
        "🏭 Filter by Sector",
        sector_options,
        key="canslim_sector_filter",
    )
    filtered_results = (
        results if selected_sector == "ALL"
        else [r for r in results if r.get('sector') == selected_sector]
    )

    # ── Ranked summary table ──────────────────────────────────────────────
    st.subheader("🏆 Ranked Scoring Table")

    rows, rank, prev = [], 1, None
    for i, r in enumerate(filtered_results):
        if r['score'] != prev:
            rank = i + 1
        rows.append({
            'Rank':               rank,
            'Ticker':             r['symbol'],
            'Sector':             r.get('sector') or '—',
            'Industry':           r.get('industry') or '—',
            'Total Score (/100)': r['score'],
            'Criteria Met':       f"{r['criteria_met']} / 10",
            'Data Gaps':          r['data_gaps'],
        })
        prev = r['score']

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Per-ticker detailed breakdown ─────────────────────────────────────
    st.subheader("🔍 Detailed Breakdown")
    st.caption("Expand any ticker to see its full metrics, scoring, and quarterly trend.")

    for r in filtered_results:
        sym  = r['symbol']
        lq   = r['q_dates'][0] if r['q_dates'] else "—"
        la   = r['a_dates'][0] if r['a_dates'] else "—"
        m    = r['metrics']

        with st.expander(f"{sym}  —  {r['score']}/100   |  {r['criteria_met']}/10 criteria met  |  {r['data_gaps']} data gap(s)"):

            # Metrics table
            st.markdown("**Computed Metrics**")
            st.dataframe(pd.DataFrame([
                ("#1–4",  "EPS YoY Growth Q4 / Q3 / Q2 / Q1",
                 f"{_pct(m.get('eps_gr_q4'))} / {_pct(m.get('eps_gr_q3'))} / {_pct(m.get('eps_gr_q2'))} / {_pct(m.get('eps_gr_q1'))}", lq),
                ("#3",   "EPS Acceleration (3 Qtrs)",      _fmt_criterion("Acceleration", m.get('eps_accel')), lq),
                ("#4",   "EPS 3Y Avg Annual Growth",        _pct(m.get('eps_3y_avg')), la),
                ("#5",   "EPS 5Y Avg Annual Growth",        _pct(m.get('eps_5y_avg')), la),
                ("#6",   "Revenue YoY Growth Q4",           _pct(m.get('rev_gr_q4')), lq),
                ("#7",   "Avg Revenue Growth (Q2–Q4)",      _pct(m.get('rev_3q_avg')), lq),
                ("#8",   "Pretax Income Annual Growth",     _pct(m.get('pretax_ann_gr')), la),
                ("#9",   "Net Income Annual Growth",        _pct(m.get('ni_ann_gr')), la),
                ("#10a", "ROE Q4 (latest)",                 _pct(m.get('roe_q4')), lq),
                ("#10b", "ROE YoY Growth",                  _pct(m.get('roe_yoy_gr')), lq),
                ("#11",  "Institutional Ownership",         _pct(m.get('inst_pct')), "latest"),
            ], columns=["#", "Metric", "Value", "Period"]),
            use_container_width=True, hide_index=True)

            # Scoring table
            st.markdown("**Scoring Criteria**")
            score_rows = [{'Criterion': sd['Criterion'],
                           'Value':     _fmt_criterion(sd['Criterion'], sd['Value']),
                           'Threshold': sd['Threshold'],
                           'Points':    sd['Points'],
                           'Status':    _traffic(sd['Met'])}
                          for sd in r['score_details']]
            st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

            st.metric(f"Total CANSLIM Score: {sym}", f"{r['score']} / 100")

            # Quarterly trend
            st.markdown("**Last 4 Quarters — YoY Growth Trend**")
            q_labels = ["Q1 (oldest)", "Q2", "Q3", "Q4 (latest)"]
            trend = [{'Quarter':     q_labels[i],
                      'EPS YoY':     _pct(r['eps_gr_series'][i] if i < len(r['eps_gr_series']) else None),
                      'Revenue YoY': _pct(r['rev_gr_series'][i] if i < len(r['rev_gr_series']) else None)}
                     for i in range(4)]
            st.dataframe(pd.DataFrame(trend), use_container_width=True, hide_index=True)

            if r['errors']:
                st.caption("⚠️ Data gaps: " + " · ".join(r['errors']))

    st.info("Expand any ticker above to see its full CANSLIM breakdown.")

else:
    if not run_btn:
        st.info("👆 Enter tickers in the sidebar and click **Run CANSLIM Analysis** to start.")

    with st.expander("📖 About CANSLIM"):
        st.markdown("""
        **CANSLIM** is a growth-stock selection methodology developed by William O'Neil.
        Each letter stands for a key criterion:

        | Letter | Stands For | This System Measures |
        |--------|------------|----------------------|
        | **C** | Current quarterly earnings | EPS YoY growth > 20% |
        | **A** | Annual earnings growth | 3Y & 5Y avg EPS growth > 25% |
        | **N** | New products / highs | *(not scored — requires qualitative input)* |
        | **S** | Supply & demand | *(not scored here)* |
        | **L** | Leader or laggard | *(not scored here)* |
        | **I** | Institutional sponsorship | Institutional ownership > 35% |
        | **M** | Market direction | *(not scored — macro context)* |

        This tool scores the **quantifiable fundamentals**: EPS acceleration, revenue growth,
        income growth, ROE momentum, and institutional ownership — 10 criteria × 10 points = 100 max.

        > **Disclaimer:** Informational and educational only. Not investment advice.
        """)
