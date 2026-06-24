"""
CANSLIM Scoring Module
Fetches fundamental data via yfinance and scores each ticker on the
10-criterion CANSLIM methodology (maximum 100 points).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings

warnings.filterwarnings('ignore')

FMP_BASE = "https://financialmodelingprep.com/stable"


# ============================================================================
# COUNTRY / EXCHANGE REFERENCE DATA
# ============================================================================

# Country → list of (exchange_code, display_label) tuples
COUNTRY_EXCHANGES = {
    "Saudi Arabia":     [("SAU",      "Tadawul (SAU)")],
    "United States":    [("NASDAQ",   "NASDAQ"), ("NYSE", "NYSE"), ("AMEX", "AMEX")],
    "United Kingdom":   [("LSE",      "London Stock Exchange (LSE)")],
    "Germany":          [("ETR",      "XETRA (ETR)"), ("FRA", "Frankfurt (FRA)")],
    "France":           [("EURONEXT", "Euronext Paris")],
    "Japan":            [("TYO",      "Tokyo Stock Exchange (TYO)")],
    "Hong Kong":        [("HKSE",     "Hong Kong Stock Exchange (HKSE)")],
    "China":            [("SHH",      "Shanghai (SHH)"), ("SHZ", "Shenzhen (SHZ)")],
    "India":            [("NSE",      "NSE India"), ("BSE", "BSE India")],
    "Canada":           [("TSX",      "Toronto Stock Exchange (TSX)"), ("TSXV", "TSX Venture (TSXV)")],
    "Australia":        [("ASX",      "Australian Securities Exchange (ASX)")],
    "South Korea":      [("KSC",      "KOSPI (KSC)"), ("KOE", "KOSDAQ (KOE)")],
    "Brazil":           [("BVMF",     "B3 - Brasil Bolsa Balcão (BVMF)")],
    "UAE":              [("ADX",      "Abu Dhabi Securities Exchange (ADX)"), ("DFM", "Dubai Financial Market (DFM)")],
    "Qatar":            [("QSE",      "Qatar Stock Exchange (QSE)")],
    "Kuwait":           [("KSE",      "Kuwait Stock Exchange (KSE)")],
    "Bahrain":          [("BHB",      "Bahrain Bourse (BHB)")],
    "Egypt":            [("EGX",      "Egyptian Exchange (EGX)")],
    "Switzerland":      [("SIX",      "SIX Swiss Exchange")],
    "Netherlands":      [("AMS",      "Euronext Amsterdam")],
    "Sweden":           [("STO",      "Nasdaq Stockholm (STO)")],
    "Norway":           [("OSL",      "Oslo Stock Exchange (OSL)")],
    "Denmark":          [("CPH",      "Nasdaq Copenhagen (CPH)")],
    "Singapore":        [("SES",      "Singapore Exchange (SES)")],
    "South Africa":     [("JSE",      "Johannesburg Stock Exchange (JSE)")],
    "Mexico":           [("BMV",      "Bolsa Mexicana de Valores (BMV)")],
    "Turkey":           [("IST",      "Borsa Istanbul (IST)")],
}

# Exchange code → ticker suffix used by FMP and yfinance
EXCHANGE_SUFFIX = {
    "SAU":      ".SR",
    "LSE":      ".L",
    "ETR":      ".DE",
    "FRA":      ".F",
    "EURONEXT": ".PA",
    "TYO":      ".T",
    "HKSE":     ".HK",
    "SHH":      ".SS",
    "SHZ":      ".SZ",
    "NSE":      ".NS",
    "BSE":      ".BO",
    "TSX":      ".TO",
    "TSXV":     ".V",
    "ASX":      ".AX",
    "KSC":      ".KS",
    "KOE":      ".KQ",
    "BVMF":     ".SA",
    "ADX":      ".AE",
    "DFM":      ".DFM",
    "QSE":      ".QA",
    "KSE":      ".KW",
    "BHB":      ".BH",
    "EGX":      ".CA",
    "SIX":      ".SW",
    "AMS":      ".AS",
    "STO":      ".ST",
    "OSL":      ".OL",
    "CPH":      ".CO",
    "SES":      ".SI",
    "JSE":      ".JO",
    "BMV":      ".MX",
    "IST":      ".IS",
    # US exchanges — no suffix
    "NASDAQ":   "",
    "NYSE":     "",
    "AMEX":     "",
}


def validate_ticker_fmp(symbol, api_key):
    """
    Look up a ticker via FMP profile endpoint.
    Returns (company_name, exchange, currency) or (None, None, None) if not found.
    """
    try:
        data = _fmp_get("profile", api_key, {"symbol": symbol})
        if data and isinstance(data, list):
            p = data[0]
            return p.get("companyName"), p.get("exchangeShortName"), p.get("currency")
    except Exception:
        pass
    return None, None, None


def format_ticker(raw_ticker, exchange_code):
    """
    Apply the correct suffix for a given exchange if not already present.
    e.g. '2020' + 'SAU' -> '2020.SR'
    e.g. 'AAPL' + 'NASDAQ' -> 'AAPL'
    """
    suffix = EXCHANGE_SUFFIX.get(exchange_code, "")
    raw = raw_ticker.strip().upper()
    if suffix and not raw.endswith(suffix):
        return raw + suffix
    return raw


# ============================================================================
# HELPERS
# ============================================================================

def safe_growth(current, prior):
    """
    Sign-safe YoY growth: (current − prior) / |prior|
    Returns None when prior == 0 or either value is None / NaN.
    """
    if current is None or prior is None:
        return None
    try:
        c, p = float(current), float(prior)
    except (TypeError, ValueError):
        return None
    if np.isnan(c) or np.isnan(p) or p == 0:
        return None
    return (c - p) / abs(p)


def _get_row(df, candidates):
    """Return the first row from df whose index label matches any candidate name."""
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _v(series, idx):
    """Value at position idx (0 = most-recent). Handles pandas Series and plain lists."""
    if series is None:
        return None
    vals = series if isinstance(series, (list, tuple)) else series.values
    if idx < 0 or idx >= len(vals):
        return None
    v = vals[idx]
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_canslim_data(symbol):
    """
    Fetch all raw data needed for CANSLIM scoring.
    Columns in all returned series are most-recent first (index 0 = latest quarter/year).
    """
    t = yf.Ticker(symbol)
    data = {'symbol': symbol.upper(), 'errors': []}

    # ── Quarterly income statement ───────────────────────────────────────
    try:
        qi = t.quarterly_income_stmt
        if qi is None or qi.empty:
            qi = t.quarterly_financials          # older yfinance attribute name
        if qi is not None and not qi.empty:
            data['q_eps']    = _get_row(qi, ['Diluted EPS', 'Basic EPS'])
            data['q_rev']    = _get_row(qi, ['Total Revenue', 'Revenue'])
            data['q_pretax'] = _get_row(qi, ['Pretax Income', 'Income Before Tax', 'EBIT'])
            data['q_ni']     = _get_row(qi, ['Net Income', 'Net Income Common Stockholders',
                                              'Net Income Including Noncontrolling Interests'])
            data['q_dates']  = [str(c.date()) for c in qi.columns]
        else:
            data.update(q_eps=None, q_rev=None, q_pretax=None, q_ni=None, q_dates=[])
            data['errors'].append('quarterly income statement unavailable')
    except Exception as e:
        data.update(q_eps=None, q_rev=None, q_pretax=None, q_ni=None, q_dates=[])
        data['errors'].append(f'quarterly income: {e}')

    # ── Quarterly balance sheet (for BVPS) ───────────────────────────────
    try:
        qb = t.quarterly_balance_sheet
        if qb is None or qb.empty:
            qb = t.quarterly_sheet
        if qb is not None and not qb.empty:
            equity = _get_row(qb, ['Stockholders Equity', 'Total Stockholders Equity',
                                    'Common Stock Equity', 'Total Equity Gross Minority Interest'])
            shares = _get_row(qb, ['Ordinary Shares Number', 'Share Issued',
                                    'Common Stock', 'Shares Outstanding'])
            if equity is not None and shares is not None:
                data['q_bvps'] = equity / shares.replace(0, np.nan)
            elif equity is not None:
                so = (t.info or {}).get('sharesOutstanding') or (t.info or {}).get('impliedSharesOutstanding')
                if so:
                    data['q_bvps'] = equity / float(so)
                else:
                    data['q_bvps'] = None
                    data['errors'].append('shares outstanding unavailable for BVPS')
            else:
                data['q_bvps'] = None
                data['errors'].append('stockholders equity unavailable')
        else:
            data['q_bvps'] = None
            data['errors'].append('quarterly balance sheet unavailable')
    except Exception as e:
        data['q_bvps'] = None
        data['errors'].append(f'quarterly balance sheet: {e}')

    # ── Annual income statement ───────────────────────────────────────────
    try:
        ai = t.income_stmt
        if ai is None or ai.empty:
            ai = t.financials
        if ai is not None and not ai.empty:
            data['a_eps']    = _get_row(ai, ['Diluted EPS', 'Basic EPS'])
            data['a_pretax'] = _get_row(ai, ['Pretax Income', 'Income Before Tax', 'EBIT'])
            data['a_ni']     = _get_row(ai, ['Net Income', 'Net Income Common Stockholders',
                                              'Net Income Including Noncontrolling Interests'])
            data['a_dates']  = [str(c.date()) for c in ai.columns]
        else:
            data.update(a_eps=None, a_pretax=None, a_ni=None, a_dates=[])
            data['errors'].append('annual income statement unavailable')
    except Exception as e:
        data.update(a_eps=None, a_pretax=None, a_ni=None, a_dates=[])
        data['errors'].append(f'annual income: {e}')

    # ── Institutional ownership ───────────────────────────────────────────
    try:
        info = t.info or {}
        pct = info.get('institutionPercentHeld') or info.get('heldPercentInstitutions')
        if pct is not None:
            pct = float(pct)
            data['inst_pct'] = pct / 100 if pct > 1 else pct   # normalise if in percent
        else:
            data['inst_pct'] = None
            data['errors'].append('institutional ownership unavailable')
    except Exception as e:
        data['inst_pct'] = None
        data['errors'].append(f'institutional ownership: {e}')

    return data


# ============================================================================
# CANSLIM COMPUTATION
# ============================================================================

def compute_canslim(data):
    """
    Compute all 11 CANSLIM metrics and the 10-criterion (100-point) score.
    """
    v = _v
    g = safe_growth

    m = {}   # computed metric values

    # ── Quarterly EPS growth (Q4=idx0 most-recent, Q0=idx4 year-ago) ────
    eps_gr = {}
    for label, idx in [('Q4', 0), ('Q3', 1), ('Q2', 2), ('Q1', 3)]:
        eps_gr[label] = g(v(data.get('q_eps'), idx), v(data.get('q_eps'), idx + 4))

    m['eps_gr_q4'] = eps_gr['Q4']
    m['eps_gr_q3'] = eps_gr['Q3']
    m['eps_gr_q2'] = eps_gr['Q2']
    m['eps_gr_q1'] = eps_gr['Q1']

    # ── EPS acceleration (3 consecutive quarters improving) ──────────────
    if all(eps_gr[q] is not None for q in ['Q2', 'Q3', 'Q4']):
        m['eps_accel'] = eps_gr['Q3'] > eps_gr['Q2'] and eps_gr['Q4'] > eps_gr['Q3']
    else:
        m['eps_accel'] = None

    # ── Annual EPS growth (most-recent year = idx 0) ─────────────────────
    a_eps_gr = [g(v(data.get('a_eps'), i), v(data.get('a_eps'), i + 1)) for i in range(5)]
    valid_aeg = [x for x in a_eps_gr if x is not None]
    m['eps_3y_avg'] = float(np.mean(valid_aeg[:3])) if len(valid_aeg) >= 3 else None
    m['eps_5y_avg'] = float(np.mean(valid_aeg[:5])) if len(valid_aeg) >= 5 else None

    # ── Quarterly revenue growth ─────────────────────────────────────────
    rev_gr = {}
    for label, idx in [('Q4', 0), ('Q3', 1), ('Q2', 2), ('Q1', 3)]:
        rev_gr[label] = g(v(data.get('q_rev'), idx), v(data.get('q_rev'), idx + 4))

    m['rev_gr_q4'] = rev_gr['Q4']
    valid_rev3 = [rev_gr[q] for q in ['Q2', 'Q3', 'Q4'] if rev_gr[q] is not None]
    m['rev_3q_avg'] = float(np.mean(valid_rev3)) if len(valid_rev3) == 3 else None

    # ── Annual pretax / net income growth ────────────────────────────────
    m['pretax_ann_gr'] = g(v(data.get('a_pretax'), 0), v(data.get('a_pretax'), 1))
    m['ni_ann_gr']     = g(v(data.get('a_ni'),     0), v(data.get('a_ni'),     1))

    # ── Quarterly ROE: EPS(Qn) / AvgBVPS(Qn) ────────────────────────────
    def quarterly_roe(idx):
        eps      = v(data.get('q_eps'),  idx)
        bvps_now = v(data.get('q_bvps'), idx)
        bvps_yr  = v(data.get('q_bvps'), idx + 4)
        if eps is None or bvps_now is None or bvps_yr is None:
            return None
        avg = (bvps_now + bvps_yr) / 2
        return eps / avg if avg != 0 else None

    roe_q4 = quarterly_roe(0)
    roe_q0 = quarterly_roe(4)
    m['roe_q4']     = roe_q4
    m['roe_yoy_gr'] = g(roe_q4, roe_q0)

    # ── Institutional ownership ───────────────────────────────────────────
    m['inst_pct'] = data.get('inst_pct')

    # ── Data gaps ────────────────────────────────────────────────────────
    gap_keys = ['eps_gr_q4', 'eps_accel', 'eps_3y_avg', 'eps_5y_avg',
                'rev_gr_q4', 'rev_3q_avg', 'pretax_ann_gr', 'ni_ann_gr',
                'roe_yoy_gr', 'inst_pct']
    data_gaps = sum(1 for k in gap_keys if m.get(k) is None)

    # ── Scoring ───────────────────────────────────────────────────────────
    criteria = [
        ('Recent Qtr EPS YoY Growth',    'eps_gr_q4',     m['eps_gr_q4'],     0.20, '> 20%'),
        ('EPS Acceleration (3 Qtrs)',     'eps_accel',     m['eps_accel'],     None, '= TRUE'),
        ('EPS 3Y Avg Annual Growth',      'eps_3y_avg',    m['eps_3y_avg'],    0.25, '> 25%'),
        ('EPS 5Y Avg Annual Growth',      'eps_5y_avg',    m['eps_5y_avg'],    0.25, '> 25%'),
        ('Recent Qtr Revenue YoY Growth', 'rev_gr_q4',     m['rev_gr_q4'],     0.25, '> 25%'),
        ('Avg Revenue Growth (3 Qtrs)',   'rev_3q_avg',    m['rev_3q_avg'],    0.25, '> 25%'),
        ('Pretax Income Annual Growth',   'pretax_ann_gr', m['pretax_ann_gr'], 0.15, '> 15%'),
        ('Net Income Annual Growth',      'ni_ann_gr',     m['ni_ann_gr'],     0.25, '> 25%'),
        ('ROE YoY Growth',                'roe_yoy_gr',    m['roe_yoy_gr'],    0.17, '> 17%'),
        ('Institutional Ownership',       'inst_pct',      m['inst_pct'],      0.35, '> 35%'),
    ]

    total_score = 0
    score_details = []
    for label, key, value, threshold, threshold_str in criteria:
        if key == 'eps_accel':
            met = True if value is True else (False if value is False else None)
            pts = 10 if value is True else 0
        elif value is None:
            met = None
            pts = 0
        else:
            met = value > threshold
            pts = 10 if met else 0
        total_score += pts
        score_details.append({'Criterion': label, 'Value': value,
                               'Threshold': threshold_str, 'Points': pts, 'Met': met})

    return {
        'symbol':       data['symbol'],
        'metrics':      m,
        'score':        total_score,
        'criteria_met': sum(1 for d in score_details if d['Met'] is True),
        'data_gaps':    data_gaps,
        'score_details': score_details,
        'errors':       data.get('errors', []),
        'q_dates':      data.get('q_dates', [])[:8],
        'a_dates':      data.get('a_dates', [])[:6],
        'eps_gr_series': [eps_gr.get(q) for q in ['Q1', 'Q2', 'Q3', 'Q4']],
        'rev_gr_series': [rev_gr.get(q) for q in ['Q1', 'Q2', 'Q3', 'Q4']],
    }


# ============================================================================
# FMP DATA FETCHING
# ============================================================================

def _fmp_get(endpoint, api_key, params=None):
    """Single FMP API request. Raises on HTTP error or API error message."""
    p = {'apikey': api_key}
    if params:
        p.update(params)
    r = requests.get(f"{FMP_BASE}/{endpoint}", params=p, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and 'Error Message' in data:
        raise ValueError(data['Error Message'])
    return data


def fetch_canslim_data_fmp(symbol, api_key):
    """
    Fetch all CANSLIM data from FMP stable API (financialmodelingprep.com/stable).
    BVPS = totalStockholdersEquity / outstandingShares (shares-float endpoint).
    Institutional ownership falls back to yfinance (no stable endpoint available).
    Returns data in the same format as fetch_canslim_data() (most-recent first).
    """
    sym = symbol.upper()
    data = {'symbol': sym, 'errors': []}

    # ── Quarterly income statement (12 quarters) ─────────────────────────
    try:
        q_inc = _fmp_get("income-statement", api_key,
                         {'symbol': sym, 'period': 'quarterly', 'limit': 12})
        if q_inc:
            data['q_eps']    = [d.get('epsDiluted') or d.get('eps') for d in q_inc]
            data['q_rev']    = [d.get('revenue')         for d in q_inc]
            data['q_pretax'] = [d.get('incomeBeforeTax') for d in q_inc]
            data['q_ni']     = [d.get('netIncome')       for d in q_inc]
            data['q_dates']  = [d.get('date', '')        for d in q_inc]
        else:
            data.update(q_eps=None, q_rev=None, q_pretax=None, q_ni=None, q_dates=[])
            data['errors'].append('FMP: no quarterly income data')
    except Exception as e:
        data.update(q_eps=None, q_rev=None, q_pretax=None, q_ni=None, q_dates=[])
        data['errors'].append(f'FMP quarterly income: {e}')

    # ── Quarterly BVPS: equity / shares-outstanding ───────────────────────
    # stable API has no bookValuePerShare field; compute from balance sheet
    try:
        q_bs = _fmp_get("balance-sheet-statement", api_key,
                        {'symbol': sym, 'period': 'quarterly', 'limit': 12})
        sf   = _fmp_get("shares-float", api_key, {'symbol': sym})
        shares_out = float(sf[0]['outstandingShares']) if sf else None

        if q_bs and shares_out:
            data['q_bvps'] = [
                d.get('totalStockholdersEquity') / shares_out
                if d.get('totalStockholdersEquity') else None
                for d in q_bs
            ]
        else:
            data['q_bvps'] = None
            data['errors'].append('FMP: could not compute BVPS (missing balance sheet or shares)')
    except Exception as e:
        data['q_bvps'] = None
        data['errors'].append(f'FMP BVPS: {e}')

    # ── Annual income statement (6 years) ────────────────────────────────
    try:
        a_inc = _fmp_get("income-statement", api_key,
                         {'symbol': sym, 'period': 'annual', 'limit': 6})
        if a_inc:
            data['a_eps']    = [d.get('epsDiluted') or d.get('eps') for d in a_inc]
            data['a_pretax'] = [d.get('incomeBeforeTax') for d in a_inc]
            data['a_ni']     = [d.get('netIncome')       for d in a_inc]
            data['a_dates']  = [d.get('date', '')        for d in a_inc]
        else:
            data.update(a_eps=None, a_pretax=None, a_ni=None, a_dates=[])
            data['errors'].append('FMP: no annual income data')
    except Exception as e:
        data.update(a_eps=None, a_pretax=None, a_ni=None, a_dates=[])
        data['errors'].append(f'FMP annual income: {e}')

    # ── Institutional ownership — yfinance fallback ───────────────────────
    # FMP stable API has no institutional-holder endpoint
    try:
        info = yf.Ticker(sym).info or {}
        pct = info.get('institutionPercentHeld') or info.get('heldPercentInstitutions')
        if pct is not None:
            pct = float(pct)
            data['inst_pct'] = pct / 100 if pct > 1 else pct
        else:
            data['inst_pct'] = None
            data['errors'].append('institutional ownership unavailable')
    except Exception as e:
        data['inst_pct'] = None
        data['errors'].append(f'institutional ownership: {e}')

    return data


# ============================================================================
# UNIVERSE SCORING
# ============================================================================

def _normalize_ticker(sym, fmp_api_key=None):
    """
    Normalize ticker symbol.
    Saudi Tadawul stocks are 4-digit numbers (e.g. 2020 → 2020.SR).
    Auto-appends .SR when FMP API key is present and ticker is bare digits.
    """
    if sym.isdigit() and len(sym) == 4:
        return sym + '.SR'
    return sym


def score_canslim_universe(symbols, fmp_api_key=None):
    """
    Score a list of tickers. Pass fmp_api_key to use FMP instead of yfinance.
    Returns list of result dicts sorted by score descending.
    """
    results = []
    for sym in symbols:
        sym = _normalize_ticker(sym.strip().upper(), fmp_api_key)
        if not sym:
            continue
        try:
            if fmp_api_key:
                raw = fetch_canslim_data_fmp(sym, fmp_api_key)
            else:
                raw = fetch_canslim_data(sym)
            scored = compute_canslim(raw)
        except Exception as e:
            scored = {
                'symbol': sym, 'score': 0, 'criteria_met': 0,
                'data_gaps': 10, 'errors': [str(e)],
                'metrics': {}, 'score_details': [], 'q_dates': [], 'a_dates': [],
                'eps_gr_series': [], 'rev_gr_series': [],
            }
        results.append(scored)

    results.sort(key=lambda x: x['score'], reverse=True)
    return results
