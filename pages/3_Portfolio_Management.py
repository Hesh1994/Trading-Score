"""
Portfolio Management — Efficient Frontier (Mean-Variance Optimisation)
Pulls tickers from the Scoring Dashboard and CANSLIM Dashboard,
lets you add extras, then computes optimal portfolios via Markowitz MVO.
"""

import sys, os, json
_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import datetime as _dt

st.set_page_config(
    page_title="Portfolio Management",
    page_icon="💼",
    layout="wide",
)

# ── FMP key (shared via session state / key file) ─────────────────────────────
_KEY_FILE = os.path.join(os.path.expanduser("~"), ".streamlit_fmp_key")
if 'fmp_key_loaded' not in st.session_state:
    try:
        if os.path.exists(_KEY_FILE):
            with open(_KEY_FILE) as _kf:
                _saved = json.load(_kf).get('key', '')
            if _saved:
                st.session_state['fmp_key_value'] = _saved
    except Exception:
        pass
    st.session_state['fmp_key_loaded'] = True

fmp_key = st.session_state.get('fmp_key_value', '')

# ── Page header (sticky) ──────────────────────────────────────────────────────
# The entire header is rendered as one HTML block so it can be position:sticky.
# A zero-height iframe (st.components) wires the HTML button → real st.button.
st.markdown("""
<style>
.pm-sticky {
    position: sticky;
    top: 2.75rem;
    z-index: 999;
    background: white;
    padding: 0.6rem 0 0.7rem;
    border-bottom: 3px solid black;
    margin-bottom: 1rem;
}
[data-theme="dark"] .pm-sticky { background: #0e1117; }
.pm-hdr-row { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.pm-title { font-size: 2rem; font-weight: 700; margin: 0; line-height: 1.2; }
.pm-cap { font-size: 0.8rem; color: grey; margin-top: 0.2rem; }
.pm-opt-btn {
    background: #ff4b4b;
    color: white;
    border: none;
    border-radius: 0.4rem;
    padding: 0.45rem 1.1rem;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
}
.pm-opt-btn:hover { background: #c0392b; }
</style>
<div class="pm-sticky">
  <div class="pm-hdr-row">
    <div>
      <div class="pm-title">💼 Portfolio Management</div>
      <div class="pm-cap">
        Build a portfolio from your screener tickers.
        Allocation is derived from <strong>mean-variance optimisation</strong>
        (Markowitz Efficient Frontier) using historical daily returns.
      </div>
    </div>
    <button class="pm-opt-btn" id="pm-fake-btn">📐 Optimise Portfolio</button>
  </div>
</div>
""", unsafe_allow_html=True)

# Zero-height iframe — JS reaches window.parent to wire fake → real button
import streamlit.components.v1 as _stc
_stc.html("""
<script>
(function wire(){
  var pd = window.parent.document;
  var fake = pd.getElementById('pm-fake-btn');
  if(!fake){ setTimeout(wire, 150); return; }
  // Find the real Streamlit button by label text
  var all = pd.querySelectorAll('button');
  var real = null;
  for(var i=0;i<all.length;i++){
    if(all[i].textContent.trim().indexOf('Optimise') !== -1 &&
       all[i] !== fake){ real = all[i]; break; }
  }
  if(!real){ setTimeout(wire, 150); return; }
  // Hide the real button's Streamlit wrapper
  var wrap = real.closest('[data-testid="stButton"]') || real.parentElement;
  if(wrap) wrap.style.cssText = 'height:0;overflow:hidden;margin:0;padding:0;';
  // Wire click
  fake.addEventListener('click', function(){ real.click(); });
})();
</script>
""", height=0, scrolling=False)

_pm_run = st.button("📐 Optimise Portfolio", type="primary",
                    use_container_width=False, key="pm_run_btn")

# ── Score status banners ──────────────────────────────────────────────────────
_cs_scores     = st.session_state.get('canslim_adjusted_scores', {})
_ta_scores     = st.session_state.get('ta_scores', {})
_fg_scores     = st.session_state.get('ta_fg_scores', {})
_final_scores  = st.session_state.get('ta_final_scores', {})
_wts           = st.session_state.get('ta_score_weights', {})

# Recompute final scores from components if not saved (e.g. after fresh deployment)
if not _final_scores and (_ta_scores or _cs_scores or _fg_scores):
    _wt = _wts.get('w_tech', 1.0)
    _wc = _wts.get('w_canslim', 0.0)
    _wf = _wts.get('w_fg', 0.0)
    _all_syms = set(list(_ta_scores.keys()) + list(_cs_scores.keys()) + list(_fg_scores.keys()))
    for _sym in _all_syms:
        _ws_sum, _wt_sum = 0.0, 0.0
        _tv = _ta_scores.get(_sym)
        if _tv is not None and _wt > 0:
            _ws_sum += float(_tv) * _wt; _wt_sum += _wt
        _cv = _cs_scores.get(_sym)
        if _cv is not None and _wc > 0:
            _ws_sum += float(_cv) * _wc; _wt_sum += _wc
        _fv = _fg_scores.get(_sym)
        if _fv is not None and _wf > 0:
            _ws_sum += float(_fv) * _wf; _wt_sum += _wf
        if _wt_sum == 0 and _tv is not None:
            _ws_sum, _wt_sum = float(_tv), 1.0
        if _wt_sum > 0:
            _final_scores[_sym] = round(_ws_sum / _wt_sum, 1)

_b1, _b2 = st.columns(2)
if _cs_scores:
    _b1.success(f"✅ CANSLIM scores loaded for {len(_cs_scores)} ticker(s)")
else:
    _b1.warning("⚠️ No CANSLIM scores — run the CANSLIM Dashboard first")
if _final_scores:
    _b2.success(f"✅ Final scores loaded for {len(_final_scores)} ticker(s)")
elif _ta_scores:
    _b2.success(f"✅ Technical scores loaded for {len(_ta_scores)} ticker(s)")
else:
    _b2.warning("⚠️ No scores — run the Scoring Dashboard first")

st.markdown("")

# ── Collect candidate tickers ─────────────────────────────────────────────────
_pm_canslim  = list(st.session_state.get('canslim_ticker_list', []))
_pm_ta       = list(st.session_state.get('ta_ticker_list', []))
_pm_all_pool = list(dict.fromkeys(_pm_canslim + _pm_ta))

# ── Score threshold filter ────────────────────────────────────────────────────
st.subheader("🎯 Score Threshold Filter")
_th_col1, _th_col2, _th_col3 = st.columns([2, 2, 3])
with _th_col1:
    _score_type = st.selectbox(
        "Filter by score",
        options=["Final Score (%)", "TA Score (%)", "CANSLIM Score (%)", "Fear & Greed (0–100)", "Either (TA or CANSLIM)", "Both (TA and CANSLIM)"],
        key="pm_score_type",
    )
with _th_col2:
    _threshold = st.number_input(
        "Minimum score (%)", min_value=0, max_value=100, value=40, step=5, key="pm_threshold"
    )
with _th_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"Only tickers with score **≥ {_threshold}%** will be eligible. "
        "You can still manually override below."
    )

# Build score lookup (CANSLIM normalised to %)
def _final_pct(sym):
    v = _final_scores.get(sym)
    return round(float(v), 1) if v is not None else None

def _canslim_pct(sym):
    v = _cs_scores.get(sym)
    return round(float(v), 1) if v is not None else None

def _ta_pct(sym):
    v = _ta_scores.get(sym)
    return round(float(v), 1) if v is not None else None

def _fg_val(sym):
    v = _fg_scores.get(sym)
    return round(float(v), 1) if v is not None else None

def _passes_threshold(sym):
    fs = _final_pct(sym)
    cs = _canslim_pct(sym)
    ta = _ta_pct(sym)
    fg = _fg_val(sym)
    if _score_type == "Final Score (%)":
        return fs is not None and fs >= _threshold
    elif _score_type == "TA Score (%)":
        return ta is not None and ta >= _threshold
    elif _score_type == "CANSLIM Score (%)":
        return cs is not None and cs >= _threshold
    elif _score_type == "Fear & Greed (0–100)":
        return fg is not None and fg >= _threshold
    elif _score_type == "Either (TA or CANSLIM)":
        return (ta is not None and ta >= _threshold) or (cs is not None and cs >= _threshold)
    else:  # Both
        return (ta is not None and ta >= _threshold) and (cs is not None and cs >= _threshold)

# Score summary table
if _pm_all_pool:
    _score_rows = []
    for _t in sorted(set(_pm_all_pool)):
        _fs = _final_pct(_t)
        _cs = _canslim_pct(_t)
        _ta = _ta_pct(_t)
        _fg = _fg_val(_t)
        _ok = _passes_threshold(_t)
        _score_rows.append({
            'Ticker':           _t,
            'Final Score (%)':  _fs,
            'TA Score (%)':     _ta,
            'CANSLIM (%)':      _cs,
            'Fear & Greed':     _fg,
            'Passes Filter':    '✅ Yes' if _ok else '❌ No',
        })
    st.dataframe(
        pd.DataFrame(_score_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            'Ticker':          st.column_config.TextColumn('Ticker'),
            'Final Score (%)': st.column_config.NumberColumn('Final Score %', format='%.1f'),
            'TA Score (%)':    st.column_config.NumberColumn('TA Score %', format='%.1f'),
            'CANSLIM (%)':     st.column_config.NumberColumn('CANSLIM %', format='%.1f'),
            'Fear & Greed':    st.column_config.NumberColumn('Fear & Greed', format='%.1f'),
            'Passes Filter':   st.column_config.TextColumn('Passes Filter'),
        },
    )

_pm_qualifying = [t for t in sorted(set(_pm_all_pool)) if _passes_threshold(t)]

_pm_col1, _pm_col2 = st.columns([3, 1])
with _pm_col1:
    _pm_selected = st.multiselect(
        f"Portfolio tickers — {len(_pm_qualifying)} pass the ≥{_threshold}% filter (edit freely)",
        options=sorted(set(_pm_all_pool)),
        default=_pm_qualifying,
        key="pm_ticker_sel",
    )
with _pm_col2:
    _pm_extra_raw = st.text_input(
        "Add extra ticker(s)",
        placeholder="e.g. MSFT, GOOG",
        key="pm_extra_tickers",
    )
    if _pm_extra_raw:
        for _t in [x.strip().upper() for x in _pm_extra_raw.split(",") if x.strip()]:
            if _t and _t not in _pm_selected:
                _pm_selected.append(_t)

# ── Parameters ────────────────────────────────────────────────────────────────
_pc1, _pc2, _pc3 = st.columns(3)
_pm_lookback    = _pc1.number_input("History (days)", min_value=60, max_value=1260,
                                    value=252, step=21, key="pm_lookback")
_pm_rf          = _pc2.number_input("Risk-free rate (%)", min_value=0.0, max_value=20.0,
                                    value=4.5, step=0.1, key="pm_rf") / 100
_pm_n_port      = _pc3.number_input("Frontier points", min_value=50, max_value=500,
                                    value=200, step=50, key="pm_nport")
_pm_allow_short = st.checkbox("Allow short selling", value=False, key="pm_allow_short")

if not _pm_run:
    if not _pm_all_pool:
        st.info("Add tickers via the **Scoring Dashboard** or **CANSLIM Dashboard** first, or type them in the extra tickers box above.")
    else:
        st.info(f"{len(_pm_selected)} ticker(s) selected. Adjust the parameters above then click **Optimise Portfolio**.")
    st.stop()

# ── Validation ────────────────────────────────────────────────────────────────
if len(_pm_selected) < 2:
    st.warning("Select at least 2 tickers for portfolio optimisation.")
    st.stop()

import numpy as np
try:
    from scipy.optimize import minimize
except ImportError:
    st.error("scipy is required for optimisation. Run: `pip install scipy`")
    st.stop()

# ── 1. Fetch / reuse price data ───────────────────────────────────────────────
_pm_price_cache = st.session_state.get('ta_price_data', {})
_pm_closes = {}
_missing   = []

for _sym in _pm_selected:
    if _sym in _pm_price_cache and not _pm_price_cache[_sym].empty:
        _df_sym = _pm_price_cache[_sym].copy()
        _df_sym.index = pd.to_datetime(_df_sym.index)
        _pm_closes[_sym] = _df_sym['close'].tail(int(_pm_lookback))
    else:
        _missing.append(_sym)

if _missing and fmp_key:
    _end   = _dt.date.today()
    _start = _end - _dt.timedelta(days=int(_pm_lookback * 1.5))
    try:
        from fmp_module import fetch_price_data_fmp
        for _sym in _missing:
            with st.spinner(f"Fetching price data for {_sym}…"):
                _df_m = fetch_price_data_fmp(
                    _sym, _start.strftime('%Y-%m-%d'),
                    _end.strftime('%Y-%m-%d'), fmp_key, 'daily'
                )
            if _df_m is not None and not _df_m.empty:
                _df_m.index = pd.to_datetime(_df_m.index)
                _pm_closes[_sym] = _df_m['close'].tail(int(_pm_lookback))
            else:
                st.warning(f"No price data for **{_sym}** — skipped.")
    except Exception as _fe:
        st.warning(f"FMP fetch error: {_fe}")
elif _missing:
    st.warning(
        f"No cached price data for: {', '.join(_missing)}. "
        "Run the Scoring Dashboard first to cache prices, or enter your FMP API key."
    )

# Keep only tickers with enough data
_valid = {s: v for s, v in _pm_closes.items() if len(v) >= 30}
if len(_valid) < 2:
    st.error("Need at least 2 tickers with ≥ 30 bars of price history.")
    st.stop()

# ── 2. Returns & covariance ───────────────────────────────────────────────────
_prices_df = pd.DataFrame(_valid).dropna(how='all').ffill().dropna()
_rets      = _prices_df.pct_change().dropna()
_mu        = _rets.mean() * 252          # annualised expected returns
_sigma     = _rets.cov()  * 252          # annualised covariance
_tickers   = list(_valid.keys())
_n         = len(_tickers)

# ── 3. Optimisation ───────────────────────────────────────────────────────────
_w0       = np.ones(_n) / _n
_bounds   = ((-1, 1) if _pm_allow_short else (0, 1),) * _n
_cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

def _port_ret(w): return float(np.dot(w, _mu.values))
def _port_vol(w): return float(np.sqrt(w @ _sigma.values @ w))
def _neg_sharpe(w):
    r, v = _port_ret(w), _port_vol(w)
    return -(r - _pm_rf) / v if v > 1e-9 else 0.0

with st.spinner("Optimising portfolios…"):
    _res_ms = minimize(_neg_sharpe, _w0, method='SLSQP',
                       bounds=_bounds, constraints=[_cons_sum],
                       options={'ftol': 1e-9, 'maxiter': 1000})
    _w_ms   = _res_ms.x
    _ret_ms = _port_ret(_w_ms)
    _vol_ms = _port_vol(_w_ms)

    _res_mv = minimize(_port_vol, _w0, method='SLSQP',
                       bounds=_bounds, constraints=[_cons_sum],
                       options={'ftol': 1e-9, 'maxiter': 1000})
    _w_mv   = _res_mv.x
    _ret_mv = _port_ret(_w_mv)
    _vol_mv = _port_vol(_w_mv)

    _ret_min_f = _ret_mv
    _ret_max_f = float(_mu.max())
    _targets   = np.linspace(_ret_min_f, _ret_max_f, int(_pm_n_port))
    _front_vol, _front_ret = [], []
    for _tgt in _targets:
        _cons = [_cons_sum, {'type': 'eq', 'fun': lambda w, t=_tgt: _port_ret(w) - t}]
        _r    = minimize(_port_vol, _w0, method='SLSQP',
                         bounds=_bounds, constraints=_cons,
                         options={'ftol': 1e-9, 'maxiter': 500})
        if _r.success:
            _front_vol.append(_port_vol(_r.x) * 100)
            _front_ret.append(_port_ret(_r.x) * 100)

_asset_vol = [float(np.sqrt(_sigma.loc[t, t])) * 100 for t in _tickers]
_asset_ret = [float(_mu[t]) * 100 for t in _tickers]

# ── Holding period returns & equity curves ────────────────────────────────────
_equal_w        = np.ones(_n) / _n
_port_rets_ms   = _rets.values @ _w_ms
_port_rets_mv   = _rets.values @ _w_mv
_port_rets_eq   = _rets.values @ _equal_w

_cum_ms  = np.cumprod(1 + _port_rets_ms)
_cum_mv  = np.cumprod(1 + _port_rets_mv)
_cum_eq  = np.cumprod(1 + _port_rets_eq)

_hpr_ms  = float(_cum_ms[-1] - 1) * 100
_hpr_mv  = float(_cum_mv[-1] - 1) * 100
_hpr_eq  = float(_cum_eq[-1] - 1) * 100

_sharpe_ms = (_ret_ms - _pm_rf) / _vol_ms if _vol_ms > 0 else 0.0
_sharpe_mv = (_ret_mv - _pm_rf) / _vol_mv if _vol_mv > 0 else 0.0

# ── 4. Summary metrics ────────────────────────────────────────────────────────
st.subheader("📈 Max Sharpe Portfolio")
_ms1, _ms2, _ms3, _ms4 = st.columns(4)
_ms1.metric("Annualised Return",    f"{_ret_ms*100:.2f}%")
_ms2.metric("Holding Period Return", f"{_hpr_ms:.2f}%",
            help=f"Total cumulative return over {_pm_lookback} trading days")
_ms3.metric("Annualised Volatility", f"{_vol_ms*100:.2f}%")
_ms4.metric("Sharpe Ratio",          f"{_sharpe_ms:.2f}")

st.subheader("🛡️ Min Variance Portfolio")
_mv1, _mv2, _mv3, _mv4 = st.columns(4)
_mv1.metric("Annualised Return",     f"{_ret_mv*100:.2f}%")
_mv2.metric("Holding Period Return", f"{_hpr_mv:.2f}%",
            help=f"Total cumulative return over {_pm_lookback} trading days")
_mv3.metric("Annualised Volatility", f"{_vol_mv*100:.2f}%")
_mv4.metric("Sharpe Ratio",          f"{_sharpe_mv:.2f}")

st.markdown("")

# ── 5. Efficient Frontier chart ───────────────────────────────────────────────
import plotly.graph_objects as go

_fig_ef = go.Figure()

if _front_vol:
    _fig_ef.add_trace(go.Scatter(
        x=_front_vol, y=_front_ret,
        mode='lines', name='Efficient Frontier',
        line=dict(color='#3498db', width=2.5),
    ))

_fig_ef.add_trace(go.Scatter(
    x=_asset_vol, y=_asset_ret,
    mode='markers+text', name='Assets',
    marker=dict(size=10, color='#95a5a6', symbol='circle'),
    text=_tickers, textposition='top center',
    textfont=dict(size=11),
))

_fig_ef.add_trace(go.Scatter(
    x=[_vol_ms * 100], y=[_ret_ms * 100],
    mode='markers+text', name='Max Sharpe',
    marker=dict(size=15, color='#2ecc71', symbol='star'),
    text=['Max Sharpe'], textposition='top right',
))

_fig_ef.add_trace(go.Scatter(
    x=[_vol_mv * 100], y=[_ret_mv * 100],
    mode='markers+text', name='Min Variance',
    marker=dict(size=15, color='#e74c3c', symbol='diamond'),
    text=['Min Var'], textposition='top right',
))

_fig_ef.update_layout(
    title='Efficient Frontier',
    xaxis_title='Annualised Volatility (%)',
    yaxis_title='Annualised Return (%)',
    height=520,
    legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=40, r=40, t=60, b=40),
)
_fig_ef.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
_fig_ef.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
st.plotly_chart(_fig_ef, use_container_width=True)

# ── 5b. Equity curve ──────────────────────────────────────────────────────────
st.subheader("📉 Portfolio Equity Curve")
_eq_dates = _rets.index

_fig_eq = go.Figure()
_fig_eq.add_trace(go.Scatter(
    x=_eq_dates, y=_cum_ms * 100,
    mode='lines', name='Max Sharpe',
    line=dict(color='#2ecc71', width=2),
))
_fig_eq.add_trace(go.Scatter(
    x=_eq_dates, y=_cum_mv * 100,
    mode='lines', name='Min Variance',
    line=dict(color='#e74c3c', width=2),
))
_fig_eq.add_trace(go.Scatter(
    x=_eq_dates, y=_cum_eq * 100,
    mode='lines', name='Equal Weight',
    line=dict(color='#95a5a6', width=1.5, dash='dot'),
))
# Individual assets (faint)
for _t in _tickers:
    _cum_asset = (1 + _rets[_t]).cumprod() * 100
    _fig_eq.add_trace(go.Scatter(
        x=_eq_dates, y=_cum_asset,
        mode='lines', name=_t,
        line=dict(width=1),
        opacity=0.35,
        showlegend=True,
    ))

_fig_eq.add_hline(y=100, line_dash='dash', line_color='gray', line_width=1)
_fig_eq.update_layout(
    xaxis_title='Date',
    yaxis_title='Growth of $100',
    height=460,
    legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=40, r=40, t=60, b=40),
)
_fig_eq.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
_fig_eq.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
st.plotly_chart(_fig_eq, use_container_width=True)

# ── 6. Weights & metrics table ────────────────────────────────────────────────
st.subheader("📊 Portfolio Weights & Metrics")

_wt_rows = []
for i, _t in enumerate(_tickers):
    _asset_hpr = round(float((1 + _rets[_t]).cumprod().iloc[-1] - 1) * 100, 2)
    _wt_rows.append({
        'Ticker':             _t,
        'Final Score (%)':    _final_scores.get(_t),
        'TA Score (%)':       _ta_scores.get(_t),
        'CANSLIM Score':      round(_cs_scores[_t], 1) if _t in _cs_scores else None,
        'Fear & Greed':       round(float(_fg_scores[_t]), 1) if _t in _fg_scores else None,
        'HPR (%)':            _asset_hpr,
        'Exp. Return (% pa)': round(_asset_ret[i], 2),
        'Volatility (% pa)':  round(_asset_vol[i], 2),
        'Sharpe Ratio':       round((_asset_ret[i]/100 - _pm_rf) / (_asset_vol[i]/100), 2)
                              if _asset_vol[i] > 0 else None,
        'Max Sharpe Wt (%)':  round(float(_w_ms[i]) * 100, 2),
        'Min Var Wt (%)':     round(float(_w_mv[i]) * 100, 2),
    })

st.dataframe(
    pd.DataFrame(_wt_rows),
    use_container_width=True,
    hide_index=True,
    column_config={
        'Ticker':             st.column_config.TextColumn('Ticker'),
        'Final Score (%)':    st.column_config.NumberColumn('Final Score %', format='%.1f'),
        'TA Score (%)':       st.column_config.NumberColumn('TA Score %', format='%.1f'),
        'CANSLIM Score':      st.column_config.NumberColumn('CANSLIM', format='%.1f'),
        'Fear & Greed':       st.column_config.NumberColumn('Fear & Greed', format='%.1f'),
        'HPR (%)':            st.column_config.NumberColumn('HPR %', format='%.2f',
                              help='Holding Period Return — total cumulative return over the lookback window'),
        'Exp. Return (% pa)': st.column_config.NumberColumn('Exp. Return % pa', format='%.2f'),
        'Volatility (% pa)':  st.column_config.NumberColumn('Volatility % pa', format='%.2f'),
        'Sharpe Ratio':       st.column_config.NumberColumn('Sharpe', format='%.2f'),
        'Max Sharpe Wt (%)':  st.column_config.NumberColumn('Max Sharpe Wt %', format='%.2f'),
        'Min Var Wt (%)':     st.column_config.NumberColumn('Min Var Wt %', format='%.2f'),
    },
)

# ── 7. Allocation pie charts ──────────────────────────────────────────────────
_pie1 = go.Figure(go.Pie(
    labels=_tickers, values=[max(0, w) for w in _w_ms],
    hole=0.4, textinfo='label+percent',
))
_pie1.update_layout(title='Max Sharpe Allocation', height=380,
                    margin=dict(t=50, b=10, l=10, r=10))

_pie2 = go.Figure(go.Pie(
    labels=_tickers, values=[max(0, w) for w in _w_mv],
    hole=0.4, textinfo='label+percent',
))
_pie2.update_layout(title='Min Variance Allocation', height=380,
                    margin=dict(t=50, b=10, l=10, r=10))

_pie_c1, _pie_c2 = st.columns(2)
_pie_c1.plotly_chart(_pie1, use_container_width=True)
_pie_c2.plotly_chart(_pie2, use_container_width=True)

st.caption(
    "⚠️ **Disclaimer:** Portfolio weights are based solely on historical price data using "
    "mean-variance optimisation. Past performance is not indicative of future results. "
    "This is for educational/informational purposes only — not investment advice."
)
