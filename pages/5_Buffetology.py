"""
Buffetology
Buffett-style fundamental analysis: 80 indicators across Profitability,
Growth, Cash Flow, Balance Sheet, Capital Efficiency, Shareholder
Economics, Valuation, and Economic Valuation (intrinsic value / DCF).
Data via the FMP API (same key used on the Scoring Dashboard).
"""

import sys, os

_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd

import buffetology_module as bm

try:
    from canslim_module import (fetch_fmp_exchange_tickers, fetch_fmp_available_exchanges,
                                build_country_exchange_map, fetch_ticker_sectors,
                                COUNTRY_EXCHANGES)
    _canslim_ok = True
except ImportError:
    _canslim_ok = False

st.set_page_config(
    page_title="Buffetology",
    page_icon="🎩",
    layout="wide",
)

# ── Page header ───────────────────────────────────────────────────────────────
_hdr_col, _btn_col = st.columns([4, 1])
with _hdr_col:
    st.title("🎩 Buffetology")
    st.caption(
        "80 Buffett-style fundamental indicators — profitability, growth, cash flow, "
        "balance sheet strength, capital efficiency, shareholder economics, valuation, "
        "and intrinsic value — computed from raw financial statements via FMP."
    )
_btn_col.markdown('<div style="margin-top: 1.6rem;"></div>', unsafe_allow_html=True)
_run = _btn_col.button("🧮 Calculate Metrics", type="primary",
                       use_container_width=True, key="bt_run_btn")
st.markdown('<hr style="border:none;border-top:3px solid black;margin-top:0;margin-bottom:1rem;">',
            unsafe_allow_html=True)

# ── Sidebar: configuration ────────────────────────────────────────────────────
st.sidebar.header("⚙️ Buffetology Configuration")

_known = list(dict.fromkeys(
    list(st.session_state.get('ta_ticker_list', [])) +
    list(st.session_state.get('canslim_ticker_list', []))
))
_fmp_key = st.session_state.get('fmp_key_value', '')

if not _fmp_key:
    st.sidebar.warning("⬆️ Enter an FMP API key on the Scoring Dashboard first — "
                       "fundamentals are fetched from FMP.")

# ── Stocks to analyse ───────────────────────────────────────────────────────
st.sidebar.subheader("🎯 Stock(s)")
_src = st.sidebar.radio(
    "Ticker source",
    ["From Scoring Dashboard", "Ticker Finder (new tickers)"],
    horizontal=True, key="bt_src",
)

tickers = []

if _src == "From Scoring Dashboard":
    if _known:
        tickers = st.sidebar.multiselect(
            "Tickers (imported from the Scoring Dashboard)",
            options=_known, default=_known, key="bt_ticker_ms",
        )
    else:
        st.sidebar.caption("No tickers found on the Scoring Dashboard yet — "
                           "add some there, or switch to the Ticker Finder.")

else:
    # Exact replica of the Scoring Dashboard's "Ticker Finder": Country →
    # Exchange → Load Tickers → Load Sectors → Sector filter → ticker
    # multiselect with Add Selected / Add All. Shares the same FMP
    # ticker/sector caches as the Scoring Dashboard so nothing is fetched
    # twice.
    if 'bt_ticker_list' not in st.session_state:
        st.session_state['bt_ticker_list'] = []

    if not _canslim_ok:
        st.sidebar.warning("canslim_module not found — Ticker Finder unavailable.")
    elif not _fmp_key:
        st.sidebar.caption("⬆️ Enter an FMP API key (on the Scoring Dashboard) to use the Ticker Finder.")
    else:
        @st.cache_data(ttl=3600, show_spinner=False)
        def _bt_load_exchange_map(api_key):
            raw = fetch_fmp_available_exchanges(api_key)
            if raw:
                return build_country_exchange_map(raw)
            return COUNTRY_EXCHANGES

        _fc_map = _bt_load_exchange_map(_fmp_key)

        _fc_countries = st.sidebar.multiselect(
            "Country", options=sorted(_fc_map.keys()), key="bt_finder_country",
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
                "Exchange", options=list(_fc_label_to_code.keys()), key="bt_finder_exchange",
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
            if _flc.button("📋 Load Tickers", key="bt_finder_load_btn",
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
            if _frc.button("🔄 Reload", key="bt_finder_reload_btn",
                           use_container_width=True, disabled=not _fc_loaded):
                st.session_state.pop(_fc_ck, None)
                st.session_state.pop(_fc_sck, None)
                st.rerun()

            if _fc_loaded:
                _fc_tickers = st.session_state[_fc_ck]
                st.sidebar.caption(f"{len(_fc_tickers):,} tickers loaded from FMP")

                if not _fc_sloaded:
                    if st.sidebar.button("🏭 Load Sectors", key="bt_finder_load_sectors_btn",
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
                    key="bt_finder_sector_filter", placeholder="All sectors (no filter)",
                )
                _fc_in_sec = (
                    _fc_tickers if (not _fc_sec_choice or not _fc_smap)
                    else [(s, n) for s, n in _fc_tickers if _fc_smap.get(s, "") in set(_fc_sec_choice)]
                )

                _fc_opts = [f"{s}  —  {n}" for s, n in _fc_in_sec[:1000]]
                _fc_chosen = st.sidebar.multiselect(
                    f"Select ticker ({len(_fc_in_sec):,} available)", options=_fc_opts,
                    key="bt_finder_ticker_select", placeholder="Search and select tickers…",
                )
                _fb1, _fb2 = st.sidebar.columns(2)
                if _fb1.button("➕ Add Selected", key="bt_finder_add_btn",
                              use_container_width=True, disabled=not _fc_chosen):
                    _fc_before = len(st.session_state['bt_ticker_list'])
                    for _lbl in _fc_chosen:
                        _sym = _lbl.split("  —  ")[0].strip()
                        if _sym not in st.session_state['bt_ticker_list']:
                            st.session_state['bt_ticker_list'].append(_sym)
                    _fc_added = len(st.session_state['bt_ticker_list']) - _fc_before
                    st.session_state.pop("bt_finder_ticker_select", None)
                    st.sidebar.success(f"Added {_fc_added} ticker(s)")
                    st.rerun()
                if _fb2.button("➕ Add All", key="bt_finder_add_all_btn",
                              use_container_width=True):
                    _fc_before = len(st.session_state['bt_ticker_list'])
                    for _s, _ in _fc_in_sec:
                        if _s not in st.session_state['bt_ticker_list']:
                            st.session_state['bt_ticker_list'].append(_s)
                    _fc_added = len(st.session_state['bt_ticker_list']) - _fc_before
                    st.sidebar.success(f"Added {_fc_added} ticker(s)")
                    st.rerun()
            else:
                st.sidebar.caption("Click **Load Tickers** to browse listed stocks.")

    tickers = list(st.session_state['bt_ticker_list'])
    if tickers:
        st.sidebar.caption(f"**{len(tickers)} ticker(s) selected**")
        if st.sidebar.button("🗑️ Clear list", key="bt_finder_clear_btn"):
            st.session_state['bt_ticker_list'] = []
            st.rerun()

st.sidebar.subheader("📐 Valuation Assumptions")
st.sidebar.caption("Used only for the Economic Valuation / intrinsic value section.")
discount_rate = st.sidebar.number_input(
    "Discount rate (%)", min_value=1.0, max_value=30.0, value=10.0, step=0.5,
    key="bt_discount_rate") / 100.0
growth_rate = st.sidebar.number_input(
    "Projected Owner Earnings growth (%)", min_value=-20.0, max_value=40.0, value=8.0, step=0.5,
    key="bt_growth_rate") / 100.0
terminal_growth = st.sidebar.number_input(
    "Terminal growth (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.25,
    key="bt_terminal_growth") / 100.0
projection_years = st.sidebar.number_input(
    "Projection years", min_value=3, max_value=20, value=10, step=1,
    key="bt_projection_years")
growth_lookback = st.sidebar.number_input(
    "Growth-metric lookback (years)", min_value=2, max_value=10, value=5, step=1,
    key="bt_growth_lookback",
    help="Window used for Book Value/Share, Shares Outstanding, Dividend and "
         "Retained Earnings CAGR, and for the Normalized Earnings/FCF averages.")

if terminal_growth >= discount_rate:
    st.sidebar.error("Terminal growth must be below the discount rate, or the "
                     "terminal value formula divides by zero or goes negative.")

# ── Run ───────────────────────────────────────────────────────────────────────
if not _run and 'bt_results' not in st.session_state:
    st.info("Choose ticker(s) and set the valuation assumptions in the sidebar, then press "
            "**Calculate Metrics**.")
    st.stop()

if _run:
    if not tickers:
        st.error("Select or add at least one ticker first.")
        st.stop()
    if not _fmp_key:
        st.error("Enter an FMP API key on the Scoring Dashboard first.")
        st.stop()
    if terminal_growth >= discount_rate:
        st.error("Fix the valuation assumptions (terminal growth must be below the discount rate) before running.")
        st.stop()

    results = {}
    _prog = st.progress(0, text="Fetching fundamentals…")
    for i, t in enumerate(tickers):
        try:
            results[t] = bm.compute_metrics(
                t, _fmp_key, discount_rate=discount_rate, growth_rate=growth_rate,
                terminal_growth=terminal_growth, projection_years=int(projection_years),
                growth_lookback=int(growth_lookback),
            )
        except Exception as e:
            results[t] = {"symbol": t, "errors": [str(e)], "values": {}}
        _prog.progress((i + 1) / len(tickers), text=f"{i+1}/{len(tickers)} — {t}")
    _prog.empty()

    st.session_state['bt_results'] = results

results = st.session_state['bt_results']

_errored = {t: r["errors"] for t, r in results.items() if r.get("errors")}
if _errored:
    with st.expander(f"⚠️ Data notes for {len(_errored)} ticker(s)", expanded=False):
        for t, errs in _errored.items():
            st.caption(f"**{t}**: " + "; ".join(errs))

# ── Category tables ─────────────────────────────────────────────────────────
tabs = st.tabs([f"📊 {c}" for c in bm.CATEGORIES])

for cat, tab in zip(bm.CATEGORIES, tabs):
    with tab:
        cols = bm.indicators_by_category(cat)
        rows = []
        raw_rows = []
        for t, r in results.items():
            vals = r.get("values", {})
            row = {"Ticker": t}
            raw_row = {"Ticker": t}
            for key, label in cols:
                row[label] = bm.format_value(key, vals.get(key))
                raw_row[label] = vals.get(key)
            rows.append(row)
            raw_rows.append(raw_row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     height=min(600, 38 * (len(df) + 1)))

        raw_df = pd.DataFrame(raw_rows)
        st.download_button(
            f"⬇️ Download {cat} (CSV)",
            data=raw_df.to_csv(index=False).encode('utf-8'),
            file_name=f"buffetology_{cat.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"bt_download_{cat}",
        )

# ── Combined download ────────────────────────────────────────────────────────
st.markdown("---")
_all_rows = []
for t, r in results.items():
    vals = r.get("values", {})
    row = {"Ticker": t}
    for key, _cat, label, _fmt in bm.INDICATOR_SCHEMA:
        row[label] = vals.get(key)
    _all_rows.append(row)
_all_df = pd.DataFrame(_all_rows)
st.download_button(
    "⬇️ Download all 80 indicators (CSV)",
    data=_all_df.to_csv(index=False).encode('utf-8'),
    file_name="buffetology_all_indicators.csv",
    mime="text/csv",
)

# ── Notes tab ────────────────────────────────────────────────────────────────
with st.expander("📖 Notes: methodology & assumptions"):
    st.markdown(r"""
### What this page computes

Every indicator is computed directly from annual income statement, balance
sheet, and cash flow statement line items fetched from FMP (up to 11 years,
enough for a 10-year CAGR) plus the current quote — nothing here comes from
a pre-packaged "ratios" endpoint, so every number can be traced back to the
formula that produced it.

### Key building blocks

- **EBIT** = FMP's `operatingIncome` (Revenue − Operating Expenses).
- **NOPAT** = EBIT × (1 − effective tax rate), where the effective tax rate
  is `Income Tax Expense ÷ Pretax Income` for that year (clamped to 0–60%
  to avoid one-off tax items distorting it).
- **FCF** = FMP's `freeCashFlow` field (CFO − CapEx), or computed directly
  if that field is missing.
- **Owner Earnings** = Net Income + D&A − CapEx − Required ΔNWC. Maintenance
  CapEx isn't separately disclosed by most companies, so **total CapEx is
  used as a proxy for maintenance CapEx** — a standard simplification, but
  one that understates Owner Earnings for companies investing heavily in
  growth (where growth CapEx inflates the total).
- **Invested Capital** = Operating Net Working Capital (A/R + Inventory −
  A/P) + Net PP&E.
- **Capital Employed** = Total Assets − Current Liabilities.
- **Enterprise Value** = Market Cap + Total Debt + Preferred Stock +
  Minority Interest − Cash.
- Ratios that use "Average X" (ROE, ROIC, ROCE, ROA, turnover ratios) use
  the average of the latest two fiscal years' balance-sheet figures, or the
  single latest year if only one year of data is available.

### Growth / CAGR windows

Revenue, EPS, and FCF CAGR are computed at fixed 3-, 5-, and 10-year
windows. Book Value/Share, Shares Outstanding, Dividend, and Retained
Earnings CAGR use the **"Growth-metric lookback"** window set in the
sidebar (default 5 years) instead of a fixed period, since the source
table specifies these generically as "CAGRₙ." All CAGR figures return
blank when the company doesn't have enough history, or when either
endpoint value is zero or negative (a CAGR isn't meaningful across a
sign change).

### Economic Valuation / intrinsic value

The intrinsic value is a two-stage discounted Owner Earnings model, built
exactly from the source formulas:

- Owner Earnings are projected forward for the **Projection years** you set,
  growing at the **Projected Owner Earnings growth** rate.
- A **Terminal Value** is added at the end of the projection using the
  Gordon growth formula, `OEₙ × (1+g) ÷ (r−g)`, with the **Terminal growth**
  rate.
- Both the projected cash flows and the terminal value are discounted back
  to today at the **Discount rate**.
- **Intrinsic Value/Share** divides by the latest diluted share count.
- **Margin of Safety** = 1 − (Price ÷ Intrinsic Value/Share); positive means
  the stock trades below the model's intrinsic value, negative means above.
- **Normalized Earnings/FCF** average Net Income and FCF over the lookback
  window, smoothing out one-off spikes or cyclical troughs.
- **Expected Long-Term Shareholder Return** = Owner Earnings Yield + Owner
  Earnings Growth (over the lookback window) + Net Buyback Yield. The
  source formula's "± Valuation Multiple Change" term is a bet on future
  re-rating that can't be derived from financial statements, so it's held
  at **zero** (a "no re-rating" baseline) rather than guessed at — read
  this figure as a floor, not a forecast.

This entire section is only as good as the assumptions you set in the
sidebar — small changes to the discount rate or growth rate can swing the
intrinsic value substantially. Treat it as a framework for thinking about
value, not a precise target price.

### Data caveats

- Every value depends on FMP's reported statement data; restated financials,
  unusual reporting periods, or missing lines will blank out the indicators
  that depend on them rather than showing a misleading number.
- Forward P/E requires FMP's analyst-estimates endpoint, which isn't
  available for every ticker — it shows blank when unavailable.
- Dividend Yield, Dividend Payout, and similar per-share figures use total
  dividends paid (from the cash flow statement) divided by diluted shares,
  not a per-share dividend actually declared — a reasonable approximation
  but not identical to a company's stated dividend/share.
""")
