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
    # ── Americas ──────────────────────────────────────────────────────────
    "United States":    [("NASDAQ",   "NASDAQ"),
                         ("NYSE",     "NYSE"),
                         ("AMEX",     "AMEX"),
                         ("OTC",      "OTC Markets")],
    "Canada":           [("TSX",      "Toronto Stock Exchange (TSX)"),
                         ("TSXV",     "TSX Venture (TSXV)"),
                         ("CNSX",     "Canadian Securities Exchange (CSE)")],
    "Brazil":           [("BVMF",     "B3 – Brasil Bolsa Balcão (BVMF)")],
    "Mexico":           [("BMV",      "Bolsa Mexicana de Valores (BMV)")],
    "Argentina":        [("BCBA",     "Buenos Aires Stock Exchange (BCBA)")],
    "Chile":            [("BCS",      "Bolsa de Santiago (BCS)")],
    "Colombia":         [("BVC",      "Bolsa de Valores de Colombia (BVC)")],
    "Peru":             [("BVL",      "Bolsa de Valores de Lima (BVL)")],

    # ── Europe ────────────────────────────────────────────────────────────
    "United Kingdom":   [("LSE",      "London Stock Exchange (LSE)")],
    "Germany":          [("ETR",      "XETRA (ETR)"),
                         ("FRA",      "Frankfurt (FRA)")],
    "France":           [("EURONEXT", "Euronext Paris")],
    "Netherlands":      [("AMS",      "Euronext Amsterdam")],
    "Belgium":          [("EBR",      "Euronext Brussels")],
    "Portugal":         [("ELI",      "Euronext Lisbon")],
    "Italy":            [("MIL",      "Borsa Italiana (MIL)")],
    "Spain":            [("MCE",      "Bolsa de Madrid (MCE)")],
    "Switzerland":      [("SIX",      "SIX Swiss Exchange")],
    "Sweden":           [("STO",      "Nasdaq Stockholm (STO)")],
    "Norway":           [("OSL",      "Oslo Stock Exchange (OSL)")],
    "Denmark":          [("CPH",      "Nasdaq Copenhagen (CPH)")],
    "Finland":          [("HEL",      "Nasdaq Helsinki (HEL)")],
    "Austria":          [("VIE",      "Vienna Stock Exchange (VIE)")],
    "Poland":           [("WSE",      "Warsaw Stock Exchange (WSE)")],
    "Greece":           [("ATH",      "Athens Stock Exchange (ATH)")],
    "Russia":           [("MCX",      "Moscow Exchange (MCX)")],
    "Turkey":           [("IST",      "Borsa Istanbul (IST)")],
    "Czech Republic":   [("PRA",      "Prague Stock Exchange (PRA)")],
    "Hungary":          [("BUD",      "Budapest Stock Exchange (BUD)")],
    "Romania":          [("BVB",      "Bucharest Stock Exchange (BVB)")],

    # ── Asia Pacific ──────────────────────────────────────────────────────
    "Japan":            [("TYO",      "Tokyo Stock Exchange (TYO)")],
    "China":            [("SHH",      "Shanghai Stock Exchange (SHH)"),
                         ("SHZ",      "Shenzhen Stock Exchange (SHZ)")],
    "Hong Kong":        [("HKSE",     "Hong Kong Stock Exchange (HKSE)")],
    "South Korea":      [("KSC",      "KOSPI (KSC)"),
                         ("KOE",      "KOSDAQ (KOE)")],
    "India":            [("NSE",      "NSE India"),
                         ("BSE",      "BSE India")],
    "Australia":        [("ASX",      "Australian Securities Exchange (ASX)")],
    "New Zealand":      [("NZX",      "New Zealand Exchange (NZX)")],
    "Singapore":        [("SES",      "Singapore Exchange (SES)")],
    "Malaysia":         [("KLSE",     "Bursa Malaysia (KLSE)")],
    "Thailand":         [("SET",      "Stock Exchange of Thailand (SET)")],
    "Indonesia":        [("IDX",      "Indonesia Stock Exchange (IDX)")],
    "Philippines":      [("PSE",      "Philippine Stock Exchange (PSE)")],
    "Taiwan":           [("TAI",      "Taiwan Stock Exchange (TAI)"),
                         ("TWO",      "Taipei Exchange (TWO)")],
    "Vietnam":          [("HNX",      "Hanoi Stock Exchange (HNX)"),
                         ("HOSE",     "Ho Chi Minh Stock Exchange (HOSE)")],

    # ── Middle East ───────────────────────────────────────────────────────
    "Saudi Arabia":     [("SAU",      "Tadawul (SAU)")],
    "UAE":              [("ADX",      "Abu Dhabi Securities Exchange (ADX)"),
                         ("DFM",      "Dubai Financial Market (DFM)")],
    "Qatar":            [("QSE",      "Qatar Stock Exchange (QSE)")],
    "Kuwait":           [("KSE",      "Kuwait Stock Exchange (KSE)")],
    "Bahrain":          [("BHB",      "Bahrain Bourse (BHB)")],
    "Oman":             [("MSM",      "Muscat Securities Market (MSM)")],
    "Jordan":           [("ASE",      "Amman Stock Exchange (ASE)")],
    "Israel":           [("TASE",     "Tel Aviv Stock Exchange (TASE)")],

    # ── Africa ────────────────────────────────────────────────────────────
    "Egypt":            [("EGX",      "Egyptian Exchange (EGX)")],
    "South Africa":     [("JSE",      "Johannesburg Stock Exchange (JSE)")],
    "Morocco":          [("CAS",      "Casablanca Stock Exchange (CAS)")],
    "Nigeria":          [("NGM",      "Nigerian Exchange (NGM)")],
    "Kenya":            [("NSE_KE",   "Nairobi Securities Exchange (NSE)")],
}

# Exchange code → ticker suffix for FMP / yfinance
EXCHANGE_SUFFIX = {
    # Americas
    "NASDAQ":   "", "NYSE":  "", "AMEX":  "", "OTC":  "",
    "TSX":      ".TO", "TSXV": ".V",  "CNSX":  ".CN",
    "BVMF":     ".SA", "BMV":  ".MX", "BCBA":  ".BA",
    "BCS":      ".SN", "BVC":  ".CL", "BVL":   ".LM",
    # Europe
    "LSE":      ".L",  "ETR":  ".DE", "FRA":   ".F",
    "EURONEXT": ".PA", "AMS":  ".AS", "EBR":   ".BR",
    "ELI":      ".LS", "MIL":  ".MI", "MCE":   ".MC",
    "SIX":      ".SW", "STO":  ".ST", "OSL":   ".OL",
    "CPH":      ".CO", "HEL":  ".HE", "VIE":   ".VI",
    "WSE":      ".WA", "ATH":  ".AT", "MCX":   ".ME",
    "IST":      ".IS", "PRA":  ".PR", "BUD":   ".BD",
    "BVB":      ".RO",
    # Asia Pacific
    "TYO":      ".T",  "SHH":  ".SS", "SHZ":   ".SZ",
    "HKSE":     ".HK", "KSC":  ".KS", "KOE":   ".KQ",
    "NSE":      ".NS", "BSE":  ".BO", "ASX":   ".AX",
    "NZX":      ".NZ", "SES":  ".SI", "KLSE":  ".KL",
    "SET":      ".BK", "IDX":  ".JK", "PSE":   ".PS",
    "TAI":      ".TW", "TWO":  ".TWO","HNX":   ".HN",
    "HOSE":     ".VN",
    # Middle East
    "SAU":      ".SR", "ADX":  ".AD", "DFM":   ".DU",
    "QSE":      ".QA", "KSE":  ".KW", "BHB":   ".BH",
    "MSM":      ".OM", "ASE":  ".AM", "TASE":  ".TA",
    # Africa
    "EGX":      ".CA", "JSE":  ".JO", "CAS":   ".CS",
    "NGM":      ".LG", "NSE_KE": ".NR",
}

# Exchange code → FMP screener exchange parameter value (where it differs from our code)
FMP_EXCHANGE_CODE = {
    "ETR":    "XETRA",
    "CNSX":   "CSE",
    "HKSE":   "HKEX",
    "NSE_KE": "NSE",
}


def validate_ticker_fmp(symbol, api_key):
    """
    Look up a ticker via FMP.
    Tries /profile first, then falls back to /search for exchanges (e.g. EGX)
    that profile doesn't always cover.
    Returns (company_name, exchange, currency) or (None, None, None) if not found.
    """
    # 1. Profile — fastest, works for most exchanges
    try:
        data = _fmp_get("profile", api_key, {"symbol": symbol})
        if data and isinstance(data, list) and data[0].get("companyName"):
            p = data[0]
            return p.get("companyName"), p.get("exchangeShortName"), p.get("currency")
    except Exception:
        pass

    # 2. Search — catches listings that profile misses (e.g. EGX, smaller exchanges)
    try:
        results = _fmp_get("search", api_key, {"query": symbol, "limit": 10})
        if results and isinstance(results, list):
            sym_upper = symbol.upper()
            # Prefer an exact symbol match
            for item in results:
                if item.get("symbol", "").upper() == sym_upper:
                    return (item.get("name"),
                            item.get("exchangeShortName") or item.get("stockExchange"),
                            item.get("currency"))
            # Fall back to first result
            p = results[0]
            return (p.get("name"),
                    p.get("exchangeShortName") or p.get("stockExchange"),
                    p.get("currency"))
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

    # ── Sector / industry ────────────────────────────────────────────────
    try:
        info = t.info or {}
        data['sector']   = info.get('sector')   or ''
        data['industry'] = info.get('industry') or ''
    except Exception:
        data['sector'] = data['industry'] = ''

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
        'sector':       data.get('sector')   or '',
        'industry':     data.get('industry') or '',
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
# FMP PRICE DATA
# ============================================================================

def fetch_price_data_fmp(symbol, start_date, end_date, api_key, interval='daily'):
    """
    Fetch OHLCV data for one symbol from FMP stable API.

    Parameters
    ----------
    symbol     : str   e.g. 'AAPL' or '2222.SR'
    start_date : str or datetime.date  e.g. '2023-01-01'
    end_date   : str or datetime.date
    api_key    : str
    interval   : 'daily' | 'weekly' | 'monthly'

    Returns
    -------
    pandas.DataFrame with DatetimeIndex and lowercase columns:
        open  high  low  close  adjclose  volume
    Returns None on failure.
    """
    try:
        data = requests.get(
            f"{FMP_BASE}/historical-price-eod/full",
            params={
                'symbol':  symbol,
                'from':    str(start_date),
                'to':      str(end_date),
                'apikey':  api_key,
            },
            timeout=20,
        )
        data.raise_for_status()
        payload = data.json()
    except Exception:
        return None

    # Response may be a dict with 'historical' key, or a list directly
    if isinstance(payload, dict):
        if 'Error Message' in payload:
            return None
        rows = payload.get('historical') or payload.get('data') or []
    elif isinstance(payload, list):
        rows = payload
    else:
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # Normalise column names
    df.columns = df.columns.str.lower()
    rename = {'adjclose': 'adjclose', 'adj close': 'adjclose',
              'adj_close': 'adjclose', 'adjusted_close': 'adjclose'}
    df = df.rename(columns=rename)

    keep = [c for c in ['open', 'high', 'low', 'close', 'adjclose', 'volume'] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors='coerce')

    if 'close' not in df.columns or df.empty:
        return None

    # Resample for weekly / monthly
    if interval == 'weekly':
        df = df.resample('W').agg({
            'open':     'first',
            'high':     'max',
            'low':      'min',
            'close':    'last',
            'adjclose': 'last',
            'volume':   'sum',
        }).dropna(subset=['close'])
    elif interval == 'monthly':
        df = df.resample('ME').agg({
            'open':     'first',
            'high':     'max',
            'low':      'min',
            'close':    'last',
            'adjclose': 'last',
            'volume':   'sum',
        }).dropna(subset=['close'])

    return df if not df.empty else None


def fetch_price_universe_fmp(symbols, start_date, end_date, api_key, interval='daily'):
    """
    Fetch OHLCV for a list of symbols.
    Returns dict {symbol: DataFrame} — symbols with no data are omitted.
    """
    result = {}
    for sym in symbols:
        df = fetch_price_data_fmp(sym, start_date, end_date, api_key, interval)
        if df is not None and not df.empty:
            result[sym] = df
    return result


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


def fetch_ticker_sectors(symbol_list, api_key, progress_cb=None):
    """
    Fetch sector for every symbol via individual FMP /stable/profile calls.
    progress_cb(done, total) is called after each call if provided.
    Returns dict {symbol: sector_string}.
    """
    sector_map = {}
    total = len(symbol_list)
    for i, sym in enumerate(symbol_list):
        try:
            profile = _fmp_get("profile", api_key, {"symbol": sym})
            if isinstance(profile, list) and profile:
                p = profile[0]
            elif isinstance(profile, dict) and profile:
                p = profile
            else:
                p = {}
            sector_map[sym] = p.get("sector") or ""
        except Exception:
            sector_map[sym] = ""
        if progress_cb:
            progress_cb(i + 1, total)
    return sector_map


def fetch_fmp_exchange_tickers(exchange_code, api_key, limit=5000):
    """
    Return available tickers for an exchange as a sorted list of
    (symbol, company_name) tuples.

    Uses /stable/stock-list (90k records, symbol + companyName) and
    filters client-side by ticker suffix:
      - Exchanges with a suffix (e.g. .SR, .CA): keep tickers ending with that suffix.
      - US exchanges (no suffix): keep tickers that contain no dot.

    Raises RuntimeError with a diagnostic message on failure.
    """
    try:
        data = _fmp_get("stock-list", api_key, {})
    except Exception as e:
        raise RuntimeError(f"Failed to fetch FMP stock list: {e}")

    if not data or not isinstance(data, list):
        raise RuntimeError("FMP stock-list returned no data.")

    # Special code: return the full global list with no filtering
    if exchange_code == "__ALL__":
        matches = [
            (r["symbol"], r.get("companyName") or r["symbol"])
            for r in data if r.get("symbol")
        ]
        return sorted(matches, key=lambda x: x[0])[:limit]

    suffix = EXCHANGE_SUFFIX.get(exchange_code, None)

    if suffix:
        # e.g. .SR for Saudi, .CA for Egypt, .DE for Germany
        matches = [
            (r["symbol"], r.get("companyName") or r["symbol"])
            for r in data
            if r.get("symbol", "").endswith(suffix)
        ]
    else:
        # US / no-suffix exchanges — exclude any ticker that contains a dot
        matches = [
            (r["symbol"], r.get("companyName") or r["symbol"])
            for r in data
            if r.get("symbol") and "." not in r["symbol"]
        ]

    if not matches:
        raise RuntimeError(
            f"No tickers found for exchange '{exchange_code}' "
            f"(suffix='{suffix or 'none'}') in FMP stock-list."
        )

    return sorted(matches, key=lambda x: x[0])[:limit]


def _resolve_fmp_symbol(symbol, api_key):
    """
    Use FMP search to find the canonical symbol FMP uses for this ticker.
    Handles cases where the user supplies 'ABUK.CA' but FMP stores it differently.
    Returns the resolved FMP symbol, or the original if search finds nothing.
    """
    try:
        results = _fmp_get("search", api_key, {"query": symbol, "limit": 10})
        if results and isinstance(results, list):
            sym_upper = symbol.upper()
            base      = sym_upper.split(".")[0]
            # 1. Exact match
            for item in results:
                if item.get("symbol", "").upper() == sym_upper:
                    return item["symbol"]
            # 2. Same base ticker, any suffix (e.g. ABUK.CA → ABUK or ABUK.EGX)
            for item in results:
                if item.get("symbol", "").upper().split(".")[0] == base:
                    return item["symbol"]
    except Exception:
        pass
    return symbol


def fetch_canslim_data_fmp(symbol, api_key):
    """
    Fetch all CANSLIM data from FMP stable API (financialmodelingprep.com/stable).
    Resolves the canonical FMP symbol via search before fetching, so suffix
    mismatches (e.g. ABUK.CA vs ABUK) are corrected automatically.
    BVPS = totalStockholdersEquity / outstandingShares (shares-float endpoint).
    Institutional ownership via yfinance (no FMP stable endpoint available).
    Returns data in the same format as fetch_canslim_data() (most-recent first).
    """
    sym = _resolve_fmp_symbol(symbol.upper(), api_key)
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

    # ── Sector / industry ────────────────────────────────────────────────
    try:
        profile = _fmp_get("profile", api_key, {'symbol': sym})
        if isinstance(profile, list) and profile:
            p = profile[0]
        elif isinstance(profile, dict):
            p = profile
        else:
            p = {}
        data['sector']   = p.get('sector')   or ''
        data['industry'] = p.get('industry') or ''
    except Exception:
        data['sector'] = data['industry'] = ''

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
                'sector': '', 'industry': '',
            }
        results.append(scored)

    results.sort(key=lambda x: x['score'], reverse=True)
    return results
