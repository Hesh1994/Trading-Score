"""
Buffetology
Buffett-style fundamental analysis: 80 indicators across Profitability,
Growth, Cash Flow, Balance Sheet, Capital Efficiency, Shareholder
Economics, Valuation, and Economic Valuation (intrinsic value / DCF).
Data via the FMP API (same key used on the Scoring Dashboard).
"""

import sys, os, json

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

# ── Acceleration ─────────────────────────────────────────────────────────────
# Whether each ratio's own period-over-period growth rate is itself rising
# every step across the chosen window (not just growing, but growing
# faster each period). Computed from a historical series of each ratio's
# value — using the ACTUAL share price as of each past period, not today's.
st.sidebar.subheader("🚀 Acceleration")
st.sidebar.caption("Flags a ratio green in the tables below when its own growth "
                   "rate rose every period across this window — e.g. P/E growth "
                   "of 10% → 20% → 30% is accelerating; 10% → 20% → 15% is not.")

_af1, _af2 = st.sidebar.columns(2)
_accel_annual = _af1.checkbox("Annual", value=st.session_state.get('bt_accel_annual', True),
                              key="bt_accel_annual")
_accel_quarterly = _af2.checkbox("Quarterly", value=st.session_state.get('bt_accel_quarterly', False),
                                 key="bt_accel_quarterly")
if _accel_annual and _accel_quarterly:
    st.sidebar.warning("Both checked — using Quarterly.")
_accel_frequency = "quarter" if _accel_quarterly else "annual"

_accel_interval = st.sidebar.number_input(
    "Interval (last N quarters or N years)", min_value=3, max_value=40, value=4, step=1,
    key="bt_accel_interval",
    help="E.g. 4 with Quarterly = last 4 quarters; 5 with Annual = last 5 years. "
         "Needs at least 3 periods to judge a trend.")

_accel_checked = st.sidebar.checkbox("🚀 Calculate Acceleration", key="bt_accel_run_cb")
# Fire once on the unchecked -> checked transition, not on every rerun the
# checkbox happens to still be on (e.g. from clicking something else in the
# sidebar) — otherwise every unrelated interaction would re-trigger the
# fundamentals + price-history refetch for every ticker.
_accel_run = _accel_checked and not st.session_state.get('_bt_accel_was_checked', False)
st.session_state['_bt_accel_was_checked'] = _accel_checked

# ── Indicators & Criteria ────────────────────────────────────────────────────
# One expander per main category; each indicator inside gets a checkbox
# (whether it appears at all) and, once checked, an operator + threshold
# that defines its pass/fail criterion. Kept in session_state across reruns,
# and can be saved to disk so the same selection and thresholds come back
# the next time this page is opened (a new browser session starts a fresh
# session_state, so the on-disk copy is what survives that).
_CRITERIA_FILE = os.path.join(os.path.expanduser("~"), ".buffetology_criteria.json")


def _default_criteria():
    return {key: {'enabled': False, 'operator': '>=', 'threshold': 0.0}
           for key, _cat, _label, _fmt in bm.INDICATOR_SCHEMA}


def _load_saved_criteria():
    try:
        if os.path.exists(_CRITERIA_FILE):
            with open(_CRITERIA_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return None


st.sidebar.subheader("📋 Indicators & Criteria")
st.sidebar.caption("Check the indicators you want to see, and set a pass/fail "
                   "threshold for each.")

if 'bt_criteria_config' not in st.session_state:
    _defaults = _default_criteria()
    _saved = _load_saved_criteria()
    if _saved:
        for key, cfg in _saved.items():
            if key in _defaults and isinstance(cfg, dict):
                _defaults[key].update(cfg)
    st.session_state['bt_criteria_config'] = _defaults
else:
    for key, _cat, _label, _fmt in bm.INDICATOR_SCHEMA:
        st.session_state['bt_criteria_config'].setdefault(
            key, {'enabled': False, 'operator': '>=', 'threshold': 0.0})

_criteria = st.session_state['bt_criteria_config']

_sc1, _sc2 = st.sidebar.columns(2)
if _sc1.button("💾 Save selection", key="bt_save_criteria_btn", use_container_width=True):
    try:
        with open(_CRITERIA_FILE, 'w') as _f:
            json.dump(_criteria, _f)
        st.sidebar.success("Saved — loads automatically next time.")
    except Exception as _e:
        st.sidebar.error(f"Could not save: {_e}")
if _sc2.button("🗑️ Clear saved", key="bt_clear_saved_btn", use_container_width=True,
              disabled=not os.path.exists(_CRITERIA_FILE)):
    try:
        os.remove(_CRITERIA_FILE)
        st.sidebar.success("Cleared the saved selection.")
    except Exception as _e:
        st.sidebar.error(f"Could not clear: {_e}")
st.sidebar.caption("💾 Saved to disk" if os.path.exists(_CRITERIA_FILE)
                   else "Not saved yet — changes apply now but won't be here next visit.")

for _cat in bm.CATEGORIES:
    _cat_indicators = bm.indicators_by_category(_cat)
    _n_enabled = sum(1 for k, _ in _cat_indicators if _criteria[k]['enabled'])
    with st.sidebar.expander(f"{_cat} ({_n_enabled}/{len(_cat_indicators)})", expanded=False):
        _sa, _ca = st.columns(2)
        if _sa.button("✅ Select all", key=f"bt_selall_{_cat}", use_container_width=True):
            for k, _ in _cat_indicators:
                _criteria[k]['enabled'] = True
            st.rerun()
        if _ca.button("◻️ Clear all", key=f"bt_clearall_{_cat}", use_container_width=True):
            for k, _ in _cat_indicators:
                _criteria[k]['enabled'] = False
            st.rerun()

        for key, label in _cat_indicators:
            cfg = _criteria[key]
            enabled = st.checkbox(label, value=cfg['enabled'], key=f"bt_en_{key}")
            cfg['enabled'] = enabled
            if enabled:
                _oc, _tc = st.columns([1, 1.3])
                op_label = _oc.selectbox(
                    "Op", ["≥", "≤"], index=0 if cfg['operator'] == '>=' else 1,
                    key=f"bt_op_{key}", label_visibility="collapsed")
                cfg['operator'] = '>=' if op_label == "≥" else '<='
                cfg['threshold'] = _tc.number_input(
                    "Threshold", value=float(cfg['threshold']),
                    step=bm.default_step(key), key=f"bt_thr_{key}",
                    label_visibility="collapsed")

_enabled_keys = [k for k, cfg in _criteria.items() if cfg['enabled']]

# ── Acceleration run ──────────────────────────────────────────────────────────
if _accel_run:
    if not tickers:
        st.error("Select or add at least one ticker first.")
        st.stop()
    if not _fmp_key:
        st.error("Enter an FMP API key on the Scoring Dashboard first.")
        st.stop()

    _accel_series = {}
    _accel_errors = {}
    _prog2 = st.progress(0, text="Fetching historical data for acceleration…")
    for i, t in enumerate(tickers):
        try:
            _res = bm.compute_metrics_series(
                t, _fmp_key, frequency=_accel_frequency, num_points=int(_accel_interval),
                discount_rate=discount_rate, growth_rate=growth_rate,
                terminal_growth=terminal_growth, projection_years=int(projection_years),
                growth_lookback=int(growth_lookback),
            )
            _accel_series[t] = _res["series"]
            if _res["errors"]:
                _accel_errors[t] = _res["errors"]
        except Exception as e:
            _accel_errors[t] = [str(e)]
        _prog2.progress((i + 1) / len(tickers), text=f"{i+1}/{len(tickers)} — {t}")
    _prog2.empty()

    st.session_state['bt_accel_flags'] = bm.acceleration_flags(_accel_series)
    st.session_state['bt_accel_errors'] = _accel_errors
    st.session_state['bt_accel_meta'] = {'frequency': _accel_frequency, 'interval': int(_accel_interval)}
    st.success(f"Acceleration calculated over the last {int(_accel_interval)} "
              f"{'quarters' if _accel_frequency == 'quarter' else 'years'}.")

_accel_flags = st.session_state.get('bt_accel_flags', {})
_accel_meta = st.session_state.get('bt_accel_meta')
_accel_errors_saved = st.session_state.get('bt_accel_errors', {})
if _accel_errors_saved:
    with st.expander(f"⚠️ Acceleration data notes for {len(_accel_errors_saved)} ticker(s)", expanded=False):
        for t, errs in _accel_errors_saved.items():
            st.caption(f"**{t}**: " + "; ".join(errs))

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

def _cell(key, value):
    """Formatted value plus a pass/fail mark against its sidebar criterion."""
    text = bm.format_value(key, value)
    cfg = _criteria.get(key)
    if not cfg or not cfg['enabled'] or text == "-":
        return text
    ok = bm.passes(value, cfg['operator'], cfg['threshold'])
    mark = "✅" if ok else ("❌" if ok is False else "")
    return f"{text} {mark}".rstrip()


if not _enabled_keys:
    st.info("No indicators selected yet. Check the ones you want in "
            "**📋 Indicators & Criteria** in the sidebar — each shown value is "
            "marked ✅/❌ against the threshold you set there.")
else:
    # ── Summary: criteria met per ticker ─────────────────────────────────
    st.subheader("📋 Criteria Summary")
    _summary_rows = []
    for t, r in results.items():
        vals = r.get("values", {})
        _met = sum(1 for k in _enabled_keys
                   if bm.passes(vals.get(k), _criteria[k]['operator'], _criteria[k]['threshold']))
        _evaluable = sum(1 for k in _enabled_keys if vals.get(k) is not None)
        _summary_rows.append({
            "Ticker": t,
            "Criteria Met": f"{_met} / {len(_enabled_keys)}",
            "% of Selected Criteria": round(_met / len(_enabled_keys) * 100, 1),
            "Data Available For": f"{_evaluable} / {len(_enabled_keys)}",
        })
    _summary_df = pd.DataFrame(_summary_rows).sort_values(
        "% of Selected Criteria", ascending=False)
    st.dataframe(
        _summary_df, use_container_width=True, hide_index=True,
        column_config={
            "% of Selected Criteria": st.column_config.ProgressColumn(
                "% of Selected Criteria", min_value=0, max_value=100, format="%.0f%%"),
        },
    )
    _accel_caption = ("✅/❌ marks show whether a value meets the threshold you set in the "
                      "sidebar.")
    if _accel_meta:
        _accel_caption += (f" 🟩 green cells are **accelerating** over the last "
                          f"{_accel_meta['interval']} "
                          f"{'quarters' if _accel_meta['frequency'] == 'quarter' else 'years'} "
                          "(that ratio's own growth rate rose every period).")
    else:
        _accel_caption += (" Check **🚀 Calculate Acceleration** in the sidebar to also "
                          "highlight ratios whose growth rate is consistently accelerating.")
    st.caption(_accel_caption)

    # ── Category tables — only categories with a chosen indicator get a tab ──
    _active_categories = [
        cat for cat in bm.CATEGORIES
        if any(k in _enabled_keys for k, _ in bm.indicators_by_category(cat))
    ]
    tabs = st.tabs([f"📊 {c}" for c in _active_categories])

    for cat, tab in zip(_active_categories, tabs):
        with tab:
            cols = [(k, label) for k, label in bm.indicators_by_category(cat)
                   if k in _enabled_keys]

            rows = []
            raw_rows = []
            style_rows = []
            for t, r in results.items():
                vals = r.get("values", {})
                row = {"Ticker": t}
                raw_row = {"Ticker": t}
                style_row = {"Ticker": ""}
                for key, label in cols:
                    row[label] = _cell(key, vals.get(key))
                    raw_row[label] = vals.get(key)
                    style_row[label] = ("background-color: #b7f7c0"
                                        if _accel_flags.get(t, {}).get(key) else "")
                rows.append(row)
                raw_rows.append(raw_row)
                style_rows.append(style_row)

            df = pd.DataFrame(rows)
            _height = min(600, 38 * (len(df) + 1))
            if _accel_flags:
                _style_df = pd.DataFrame(style_rows, columns=df.columns, index=df.index)
                _styled = df.style.apply(lambda _df, _s=_style_df: _s, axis=None)
                st.dataframe(_styled, use_container_width=True, hide_index=True, height=_height)
            else:
                st.dataframe(df, use_container_width=True, hide_index=True, height=_height)

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

### Acceleration (sidebar)

"Acceleration" is a stronger condition than plain growth: it means a
ratio's own period-over-period **growth rate** is *itself* getting bigger
every period, not merely that the ratio is rising. Example: if a ratio's
period-over-period growth went **10% → 20% → 30%**, that's accelerating —
each step grew faster than the last. If instead it went **10% → 20% → 15%**,
growth continued but the *rate* of growth fell back, so that's **not**
accelerating.

To calculate it:

1. Pick **Annual** or **Quarterly** (checking both falls back to Quarterly).
2. Set the **Interval** — how many trailing periods to look at (e.g. 4 for
   "the last four quarters," or 5 for "the last five years"). At least 3
   periods are needed to judge a trend, since that gives 2 growth-rate
   readings to compare.
3. Check **🚀 Calculate Acceleration**. It runs once, on the moment you
   check it — leaving it checked afterward doesn't keep refetching every
   time you interact with something else in the sidebar; uncheck and
   re-check it to recalculate with new settings.

This re-fetches that many periods of financial statements *and* the
stock's actual historical daily prices, then recomputes every one of the
80 indicators as of each past period-end — using the real share price on
that date, not today's price — so historical P/E, P/B, dividend yield, and
every other price-based ratio reflect what they actually were at the time,
not a distortion from today's price. Every indicator's resulting time
series is then checked for a consistently rising growth rate.

Once calculated, any indicator's cell in the category tables below is
shaded **green** if it's accelerating over that window — on top of, not
instead of, its ✅/❌ pass/fail mark from the criteria you set. A blank
(unshaded) cell means either it isn't accelerating, or there wasn't enough
historical data to tell.

### Indicators & Criteria (sidebar)

Nothing shows up in the tables until you check it in **📋 Indicators &
Criteria**. Each category is its own expander; checking an indicator reveals
an operator (**≥** or **≤**) and a threshold number — that's its pass/fail
criterion. Every displayed value is then marked **✅** if it meets the
criterion, **❌** if it doesn't, or left unmarked if the underlying data is
missing. The **Criteria Summary** table at the top counts, per ticker, how
many of your selected criteria it satisfies — a quick way to rank several
candidates against the same Buffett-style screen. Thresholds are entirely
yours to set; nothing here is pre-loaded with an opinion about what "good"
looks like for any given indicator.

Your selection and thresholds live in the browser session while you work,
but a fresh session (reopening the app, restarting the server) starts
blank unless you press **💾 Save selection** — that writes the whole
configuration to a small file on this machine, which is then loaded
automatically the next time this page opens. **🗑️ Clear saved** deletes
that file so the next visit starts from scratch again.

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

### Formula Reference

Every indicator's exact formula, grouped by category, plus the supporting
items (Gross Profit, EBIT, etc.) used to build them.

**Profitability**

| Indicator | Formula |
|---|---|
| ROE | Net Income ÷ Average Shareholders' Equity × 100 |
| ROIC | NOPAT ÷ Average Invested Capital × 100 |
| ROCE | EBIT ÷ Average Capital Employed × 100 |
| ROA | Net Income ÷ Average Total Assets × 100 |
| Gross Margin | Gross Profit ÷ Revenue × 100 |
| Operating Margin | Operating Income ÷ Revenue × 100 |
| EBITDA Margin | EBITDA ÷ Revenue × 100 |
| Net Margin | Net Income ÷ Revenue × 100 |
| FCF Margin | FCF ÷ Revenue × 100 |
| Owner Earnings Margin | Owner Earnings ÷ Revenue × 100 |

**Growth**

| Indicator | Formula |
|---|---|
| Revenue CAGR 3Y | (Revenueₜ ÷ Revenueₜ₋₃)^(1/3) − 1 |
| Revenue CAGR 5Y | (Revenueₜ ÷ Revenueₜ₋₅)^(1/5) − 1 |
| Revenue CAGR 10Y | (Revenueₜ ÷ Revenueₜ₋₁₀)^(1/10) − 1 |
| EPS CAGR 3Y | (EPSₜ ÷ EPSₜ₋₃)^(1/3) − 1 |
| EPS CAGR 5Y | (EPSₜ ÷ EPSₜ₋₅)^(1/5) − 1 |
| EPS CAGR 10Y | (EPSₜ ÷ EPSₜ₋₁₀)^(1/10) − 1 |
| FCF CAGR 3Y | (FCFₜ ÷ FCFₜ₋₃)^(1/3) − 1 |
| FCF CAGR 5Y | (FCFₜ ÷ FCFₜ₋₅)^(1/5) − 1 |
| FCF CAGR 10Y | (FCFₜ ÷ FCFₜ₋₁₀)^(1/10) − 1 |
| Book Value/Share CAGR | (BVPSₜ ÷ BVPSₜ₋ₙ)^(1/n) − 1 |

**Cash Flow**

| Indicator | Formula |
|---|---|
| CFO | Cash Flow from Operations |
| FCF | CFO − Capital Expenditures |
| FCF/Share | FCF ÷ Diluted Shares Outstanding |
| Owner Earnings | Net Income + D&A − Maintenance CapEx − Required ΔNWC |
| Owner Earnings/Share | Owner Earnings ÷ Diluted Shares Outstanding |
| FCF/Net Income | FCF ÷ Net Income × 100 |
| CFO/Net Income | CFO ÷ Net Income × 100 |
| CapEx/CFO | Capital Expenditures ÷ CFO × 100 |
| CapEx/Revenue | Capital Expenditures ÷ Revenue × 100 |
| Owner Earnings Yield | Owner Earnings ÷ Market Capitalization × 100 |

**Balance Sheet**

| Indicator | Formula |
|---|---|
| Debt/Equity | Total Debt ÷ Shareholders' Equity |
| Net Debt/Equity | (Total Debt − Cash) ÷ Equity |
| Debt/EBITDA | Total Debt ÷ EBITDA |
| Net Debt/EBITDA | (Total Debt − Cash) ÷ EBITDA |
| Debt/FCF | Total Debt ÷ FCF |
| Net Debt/FCF | (Total Debt − Cash) ÷ FCF |
| Interest Coverage | EBIT ÷ Interest Expense |
| Current Ratio | Current Assets ÷ Current Liabilities |
| Quick Ratio | (Cash + Marketable Securities + A/R) ÷ Current Liabilities |
| Cash/Assets | Cash & Equivalents ÷ Total Assets × 100 |

**Capital Efficiency**

| Indicator | Formula |
|---|---|
| Asset Turnover | Revenue ÷ Average Total Assets |
| Working Capital Turnover | Revenue ÷ Average Operating Working Capital |
| Inventory Turnover | COGS ÷ Average Inventory |
| Receivables Turnover | Revenue ÷ Average Accounts Receivable |
| Payables Turnover | COGS ÷ Average Accounts Payable |
| DSO | Average A/R ÷ Revenue × 365 |
| DIO | Average Inventory ÷ COGS × 365 |
| DPO | Average A/P ÷ COGS × 365 |
| Cash Conversion Cycle | DSO + DIO − DPO |
| Incremental ROIC | ΔNOPAT ÷ ΔInvested Capital × 100 |

**Shareholder Economics**

| Indicator | Formula |
|---|---|
| Shares Outstanding CAGR | (Sharesₜ ÷ Sharesₜ₋ₙ)^(1/n) − 1 |
| EPS Growth | (EPSₜ ÷ EPSₜ₋₁) − 1 |
| FCF/Share Growth | (FCF/Shareₜ ÷ FCF/Shareₜ₋₁) − 1 |
| Dividend Yield | Dividend/Share ÷ Share Price × 100 |
| Dividend CAGR | (Dividendₜ ÷ Dividendₜ₋ₙ)^(1/n) − 1 |
| Dividend Payout | Dividends ÷ Net Income × 100 |
| FCF Payout | Dividends ÷ FCF × 100 |
| Buyback Yield | Net Share Repurchases ÷ Market Capitalization × 100 |
| Net Dilution | (Sharesₜ ÷ Sharesₜ₋₁) − 1 |
| Retained Earnings Growth | (REₜ ÷ REₜ₋ₙ)^(1/n) − 1 |

**Valuation**

| Indicator | Formula |
|---|---|
| P/E | Market Price ÷ EPS |
| Forward P/E | Current Price ÷ Forward EPS |
| P/FCF | Market Capitalization ÷ FCF |
| P/S | Market Capitalization ÷ Revenue |
| P/B | Market Capitalization ÷ Book Value |
| EV/Sales | Enterprise Value ÷ Revenue |
| EV/EBIT | Enterprise Value ÷ EBIT |
| EV/EBITDA | Enterprise Value ÷ EBITDA |
| EV/FCF | Enterprise Value ÷ FCF |
| Earnings Yield | EPS ÷ Share Price × 100 |
| FCF Yield | FCF ÷ Market Capitalization × 100 |
| Owner Earnings Yield | Owner Earnings ÷ Market Capitalization × 100 |

**Economic Valuation**

| Indicator | Formula |
|---|---|
| Normalized Earnings | Sustainable/normalized Net Income or Owner Earnings after removing abnormal items |
| Normalized FCF | Sustainable FCF after normalizing cyclicality, CapEx and working capital |
| Owner Earnings | Net Income + D&A − Maintenance CapEx − Required ΔNWC |
| Intrinsic Value | Σ[Owner Earningsₜ ÷ (1+r)ᵗ] + PV(Terminal Value) |
| Intrinsic Value/Share | Intrinsic Value ÷ Diluted Shares Outstanding |
| Price/Intrinsic Value | Current Share Price ÷ Intrinsic Value/Share |
| Margin of Safety | 1 − (Current Share Price ÷ Intrinsic Value/Share) |
| Expected Long-Term Shareholder Return | Owner Earnings Yield + Owner Earnings Growth + Net Buyback Yield ± Valuation Multiple Change |

**Supporting items** (used inside the formulas above, not shown as their own indicator)

| Item | Formula |
|---|---|
| Gross Profit | Revenue − COGS |
| EBIT | Revenue − Operating Expenses |
| NOPAT | EBIT × (1 − Effective Tax Rate) |
| FCF | CFO − CapEx |
| Operating NWC | A/R + Inventory − A/P − Other Operating Current Liabilities/Assets as appropriate |
| Invested Capital | Operating NWC + Net PP&E + Other Operating Assets − Operating Liabilities |
| Capital Employed | Total Assets − Current Liabilities |
| Enterprise Value | Market Cap + Debt + Preferred Stock + Minority Interest − Cash |
| Book Value/Share | Common Equity ÷ Diluted Shares |
| Terminal Value | OEₙ × (1+g) ÷ (r−g) |
| CAGR | (Ending Value ÷ Beginning Value)^(1/n) − 1 |

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
