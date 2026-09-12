"""
Buffetology Module
Fetches annual fundamentals via FMP and computes the 80-indicator
Buffett-style ratio set (profitability, growth, cash flow, balance sheet,
capital efficiency, shareholder economics, valuation, and economic
valuation / intrinsic value).

Pure-pandas/requests, matching the style of canslim_module.py — no ratio
library, every formula is computed explicitly from raw statement line
items so the numbers can be traced back to the formulas they're built from.
"""

import warnings

import numpy as np

from canslim_module import _fmp_get, _resolve_fmp_symbol

warnings.filterwarnings('ignore')


# ============================================================================
# INDICATOR SCHEMA — drives both computation bookkeeping and page display
# ============================================================================
# (key, category, label, format) — format is one of:
#   'pct'      -> value already a fraction/percent points; show as X.XX%
#   'x'        -> a multiple/ratio; show as X.XXx
#   'ratio'    -> a plain decimal ratio; show as X.XX
#   'money'    -> a currency amount; show with thousands separators
#   'pershare' -> a per-share currency amount; show as X.XX
#   'days'     -> a day count; show as X.X days

CATEGORIES = [
    "Profitability", "Growth", "Cash Flow", "Balance Sheet",
    "Capital Efficiency", "Shareholder Economics", "Valuation",
    "Economic Valuation",
]

INDICATOR_SCHEMA = [
    # Profitability
    ("roe", "Profitability", "ROE", "pct"),
    ("roic", "Profitability", "ROIC", "pct"),
    ("roce", "Profitability", "ROCE", "pct"),
    ("roa", "Profitability", "ROA", "pct"),
    ("gross_margin", "Profitability", "Gross Margin", "pct"),
    ("operating_margin", "Profitability", "Operating Margin", "pct"),
    ("ebitda_margin", "Profitability", "EBITDA Margin", "pct"),
    ("net_margin", "Profitability", "Net Margin", "pct"),
    ("fcf_margin", "Profitability", "FCF Margin", "pct"),
    ("owner_earnings_margin", "Profitability", "Owner Earnings Margin", "pct"),
    # Growth
    ("revenue_cagr_3y", "Growth", "Revenue CAGR 3Y", "pct"),
    ("revenue_cagr_5y", "Growth", "Revenue CAGR 5Y", "pct"),
    ("revenue_cagr_10y", "Growth", "Revenue CAGR 10Y", "pct"),
    ("eps_cagr_3y", "Growth", "EPS CAGR 3Y", "pct"),
    ("eps_cagr_5y", "Growth", "EPS CAGR 5Y", "pct"),
    ("eps_cagr_10y", "Growth", "EPS CAGR 10Y", "pct"),
    ("fcf_cagr_3y", "Growth", "FCF CAGR 3Y", "pct"),
    ("fcf_cagr_5y", "Growth", "FCF CAGR 5Y", "pct"),
    ("fcf_cagr_10y", "Growth", "FCF CAGR 10Y", "pct"),
    ("bvps_cagr", "Growth", "Book Value/Share CAGR", "pct"),
    # Cash Flow
    ("cfo", "Cash Flow", "CFO", "money"),
    ("fcf", "Cash Flow", "FCF", "money"),
    ("fcf_per_share", "Cash Flow", "FCF/Share", "pershare"),
    ("owner_earnings", "Cash Flow", "Owner Earnings", "money"),
    ("owner_earnings_per_share", "Cash Flow", "Owner Earnings/Share", "pershare"),
    ("fcf_over_ni", "Cash Flow", "FCF/Net Income", "pct"),
    ("cfo_over_ni", "Cash Flow", "CFO/Net Income", "pct"),
    ("capex_over_cfo", "Cash Flow", "CapEx/CFO", "pct"),
    ("capex_over_revenue", "Cash Flow", "CapEx/Revenue", "pct"),
    ("owner_earnings_yield", "Cash Flow", "Owner Earnings Yield", "pct"),
    # Balance Sheet
    ("debt_to_equity", "Balance Sheet", "Debt/Equity", "x"),
    ("net_debt_to_equity", "Balance Sheet", "Net Debt/Equity", "x"),
    ("debt_to_ebitda", "Balance Sheet", "Debt/EBITDA", "x"),
    ("net_debt_to_ebitda", "Balance Sheet", "Net Debt/EBITDA", "x"),
    ("debt_to_fcf", "Balance Sheet", "Debt/FCF", "x"),
    ("net_debt_to_fcf", "Balance Sheet", "Net Debt/FCF", "x"),
    ("interest_coverage", "Balance Sheet", "Interest Coverage", "x"),
    ("current_ratio", "Balance Sheet", "Current Ratio", "x"),
    ("quick_ratio", "Balance Sheet", "Quick Ratio", "x"),
    ("cash_over_assets", "Balance Sheet", "Cash/Assets", "pct"),
    # Capital Efficiency
    ("asset_turnover", "Capital Efficiency", "Asset Turnover", "x"),
    ("working_capital_turnover", "Capital Efficiency", "Working Capital Turnover", "x"),
    ("inventory_turnover", "Capital Efficiency", "Inventory Turnover", "x"),
    ("receivables_turnover", "Capital Efficiency", "Receivables Turnover", "x"),
    ("payables_turnover", "Capital Efficiency", "Payables Turnover", "x"),
    ("dso", "Capital Efficiency", "DSO", "days"),
    ("dio", "Capital Efficiency", "DIO", "days"),
    ("dpo", "Capital Efficiency", "DPO", "days"),
    ("cash_conversion_cycle", "Capital Efficiency", "Cash Conversion Cycle", "days"),
    ("incremental_roic", "Capital Efficiency", "Incremental ROIC", "pct"),
    # Shareholder Economics
    ("shares_cagr", "Shareholder Economics", "Shares Outstanding CAGR", "pct"),
    ("eps_growth", "Shareholder Economics", "EPS Growth", "pct"),
    ("fcf_per_share_growth", "Shareholder Economics", "FCF/Share Growth", "pct"),
    ("dividend_yield", "Shareholder Economics", "Dividend Yield", "pct"),
    ("dividend_cagr", "Shareholder Economics", "Dividend CAGR", "pct"),
    ("dividend_payout", "Shareholder Economics", "Dividend Payout", "pct"),
    ("fcf_payout", "Shareholder Economics", "FCF Payout", "pct"),
    ("buyback_yield", "Shareholder Economics", "Buyback Yield", "pct"),
    ("net_dilution", "Shareholder Economics", "Net Dilution", "pct"),
    ("retained_earnings_growth", "Shareholder Economics", "Retained Earnings Growth", "pct"),
    # Valuation
    ("pe", "Valuation", "P/E", "x"),
    ("forward_pe", "Valuation", "Forward P/E", "x"),
    ("p_fcf", "Valuation", "P/FCF", "x"),
    ("p_s", "Valuation", "P/S", "x"),
    ("p_b", "Valuation", "P/B", "x"),
    ("ev_sales", "Valuation", "EV/Sales", "x"),
    ("ev_ebit", "Valuation", "EV/EBIT", "x"),
    ("ev_ebitda", "Valuation", "EV/EBITDA", "x"),
    ("ev_fcf", "Valuation", "EV/FCF", "x"),
    ("earnings_yield", "Valuation", "Earnings Yield", "pct"),
    ("fcf_yield", "Valuation", "FCF Yield", "pct"),
    ("owner_earnings_yield_val", "Valuation", "Owner Earnings Yield", "pct"),
    # Economic Valuation
    ("normalized_earnings", "Economic Valuation", "Normalized Earnings", "money"),
    ("normalized_fcf", "Economic Valuation", "Normalized FCF", "money"),
    ("owner_earnings_ev", "Economic Valuation", "Owner Earnings", "money"),
    ("intrinsic_value", "Economic Valuation", "Intrinsic Value", "money"),
    ("intrinsic_value_per_share", "Economic Valuation", "Intrinsic Value/Share", "pershare"),
    ("price_to_intrinsic", "Economic Valuation", "Price/Intrinsic Value", "x"),
    ("margin_of_safety", "Economic Valuation", "Margin of Safety", "pct"),
    ("expected_return", "Economic Valuation", "Expected Long-Term Shareholder Return", "pct"),
]

INDICATOR_LABELS = {k: label for k, _cat, label, _fmt in INDICATOR_SCHEMA}
INDICATOR_FORMATS = {k: fmt for k, _cat, _label, fmt in INDICATOR_SCHEMA}


def indicators_by_category(category):
    return [(k, label) for k, cat, label, _fmt in INDICATOR_SCHEMA if cat == category]


def format_value(key, value):
    """Render one computed value as a display string, '-' if missing."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    fmt = INDICATOR_FORMATS.get(key, "ratio")
    try:
        if fmt == "pct":
            return f"{value:.2f}%"
        if fmt == "x":
            return f"{value:.2f}x"
        if fmt == "days":
            return f"{value:.1f} days"
        if fmt == "money":
            return f"{value:,.0f}"
        if fmt == "pershare":
            return f"{value:,.2f}"
        return f"{value:.2f}"
    except (TypeError, ValueError):
        return "-"


def default_step(key):
    """A sensible number_input step for this indicator's threshold field."""
    fmt = INDICATOR_FORMATS.get(key, "ratio")
    if fmt == "money":
        return 1_000_000.0
    if fmt == "days":
        return 1.0
    return 0.1


def passes(value, operator, threshold):
    """
    True/False whether `value` satisfies the criterion, None if the value
    itself is missing (can't be evaluated either way).
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value >= threshold if operator == ">=" else value <= threshold


# ============================================================================
# Small numeric helpers
# ============================================================================

def _num(x):
    """Coerce to float, or None if missing/non-numeric."""
    if x is None:
        return None
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _div(a, b):
    a, b = _num(a), _num(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _avg(a, b):
    a, b = _num(a), _num(b)
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2.0


def _cagr(end, start, years):
    end, start = _num(end), _num(start)
    if end is None or start is None or start <= 0 or end <= 0 or not years or years <= 0:
        return None
    return (end / start) ** (1.0 / years) - 1.0


def _pct(x):
    """Fraction -> percentage points (None-safe)."""
    x = _num(x)
    return None if x is None else x * 100.0


# ============================================================================
# Fetch + per-year derivation
# ============================================================================

def fetch_fundamentals(symbol, api_key, years=11):
    """
    Pull annual income statement, balance sheet, cash flow (most-recent
    first, up to `years` periods), current quote, and — best effort —
    a forward EPS estimate. Returns a dict; any piece that fails to fetch
    is an empty list/None rather than raising.
    """
    sym = _resolve_fmp_symbol(symbol.upper(), api_key)
    out = {"symbol": sym, "errors": []}

    for field, endpoint in (("income", "income-statement"),
                            ("balance", "balance-sheet-statement"),
                            ("cashflow", "cash-flow-statement")):
        try:
            rows = _fmp_get(endpoint, api_key,
                            {"symbol": sym, "period": "annual", "limit": years})
            out[field] = rows if isinstance(rows, list) else []
            if not out[field]:
                out["errors"].append(f"FMP: no annual {field} data")
        except Exception as e:
            out[field] = []
            out["errors"].append(f"FMP {field}: {e}")

    try:
        q = _fmp_get("quote", api_key, {"symbol": sym})
        out["quote"] = q[0] if isinstance(q, list) and q else (q if isinstance(q, dict) else {})
    except Exception as e:
        out["quote"] = {}
        out["errors"].append(f"FMP quote: {e}")

    out["forward_eps"] = None
    try:
        est = _fmp_get("analyst-estimates", api_key,
                       {"symbol": sym, "period": "annual", "limit": 1})
        if isinstance(est, list) and est:
            out["forward_eps"] = _num(est[0].get("estimatedEpsAvg"))
    except Exception:
        pass  # forward P/E simply stays unavailable

    return out


def _derive_year(inc, bs, cf):
    """One fiscal year's raw statement rows -> intermediate building blocks."""
    revenue = _num(inc.get("revenue"))
    cogs = _num(inc.get("costOfRevenue"))
    gross_profit = _num(inc.get("grossProfit"))
    if gross_profit is None and revenue is not None and cogs is not None:
        gross_profit = revenue - cogs

    ebit = _num(inc.get("operatingIncome"))          # Revenue − Operating Expenses
    ebitda = _num(inc.get("ebitda"))
    net_income = _num(inc.get("netIncome"))
    pretax = _num(inc.get("incomeBeforeTax"))
    tax_expense = _num(inc.get("incomeTaxExpense"))
    tax_rate = _div(tax_expense, pretax)
    if tax_rate is not None:
        tax_rate = min(max(tax_rate, 0.0), 0.60)      # clamp odd one-off tax rates
    nopat = ebit * (1 - tax_rate) if ebit is not None and tax_rate is not None else ebit

    interest_expense = _num(inc.get("interestExpense"))
    eps = _num(inc.get("epsDiluted")) or _num(inc.get("eps"))
    shares = _num(inc.get("weightedAverageShsOutDil")) or _num(inc.get("weightedAverageShsOut"))

    cfo = _num(cf.get("operatingCashFlow")) or _num(cf.get("netCashProvidedByOperatingActivities"))
    capex = abs(_num(cf.get("capitalExpenditure")) or 0.0)
    fcf = _num(cf.get("freeCashFlow"))
    if fcf is None and cfo is not None:
        fcf = cfo - capex
    d_and_a = _num(cf.get("depreciationAndAmortization")) or _num(inc.get("depreciationAndAmortization"))
    change_in_wc = _num(cf.get("changeInWorkingCapital")) or 0.0
    # Owner Earnings = Net Income + D&A − Maintenance CapEx − Required ΔNWC.
    # Maintenance CapEx isn't separately disclosed, so it's approximated by
    # total CapEx (a standard simplification); the required change in net
    # working capital is read straight off the cash-flow statement's own
    # (already cash-signed) working-capital line.
    owner_earnings = None
    if net_income is not None:
        owner_earnings = net_income + (d_and_a or 0.0) + change_in_wc - capex

    dividends_paid = abs(_num(cf.get("dividendsPaid")) or 0.0)
    buybacks = abs(_num(cf.get("commonStockRepurchased")) or 0.0)
    stock_issued = _num(cf.get("commonStockIssued")) or 0.0
    net_buybacks = buybacks - stock_issued

    total_assets = _num(bs.get("totalAssets"))
    total_current_assets = _num(bs.get("totalCurrentAssets"))
    total_current_liabilities = _num(bs.get("totalCurrentLiabilities"))
    cash = _num(bs.get("cashAndCashEquivalents"))
    short_term_inv = _num(bs.get("shortTermInvestments")) or 0.0
    receivables = _num(bs.get("netReceivables"))
    inventory = _num(bs.get("inventory"))
    payables = _num(bs.get("accountPayables"))
    total_debt = _num(bs.get("totalDebt"))
    net_debt = _num(bs.get("netDebt"))
    if net_debt is None and total_debt is not None and cash is not None:
        net_debt = total_debt - cash
    equity = _num(bs.get("totalStockholdersEquity"))
    ppe_net = _num(bs.get("propertyPlantEquipmentNet"))
    retained_earnings = _num(bs.get("retainedEarnings"))
    preferred_stock = _num(bs.get("preferredStock")) or 0.0
    minority_interest = _num(bs.get("minorityInterest")) or 0.0

    operating_nwc = None
    if receivables is not None or inventory is not None or payables is not None:
        operating_nwc = (receivables or 0.0) + (inventory or 0.0) - (payables or 0.0)
    invested_capital = None
    if operating_nwc is not None or ppe_net is not None:
        invested_capital = (operating_nwc or 0.0) + (ppe_net or 0.0)
    capital_employed = None
    if total_assets is not None and total_current_liabilities is not None:
        capital_employed = total_assets - total_current_liabilities

    bvps = _div(equity, shares)

    return dict(
        date=inc.get("date") or bs.get("date") or cf.get("date"),
        revenue=revenue, cogs=cogs, gross_profit=gross_profit,
        ebit=ebit, ebitda=ebitda, net_income=net_income, nopat=nopat,
        interest_expense=interest_expense, eps=eps, shares=shares,
        cfo=cfo, capex=capex, fcf=fcf, d_and_a=d_and_a,
        owner_earnings=owner_earnings,
        dividends_paid=dividends_paid, buybacks=buybacks,
        stock_issued=stock_issued, net_buybacks=net_buybacks,
        total_assets=total_assets, total_current_assets=total_current_assets,
        total_current_liabilities=total_current_liabilities,
        cash=cash, short_term_inv=short_term_inv, receivables=receivables,
        inventory=inventory, payables=payables, total_debt=total_debt,
        net_debt=net_debt, equity=equity, ppe_net=ppe_net,
        retained_earnings=retained_earnings, preferred_stock=preferred_stock,
        minority_interest=minority_interest,
        operating_nwc=operating_nwc, invested_capital=invested_capital,
        capital_employed=capital_employed, bvps=bvps,
    )


def derive_years(fundamentals):
    """List of per-year derived dicts, index 0 = most recent."""
    inc = fundamentals.get("income") or []
    bs = fundamentals.get("balance") or []
    cf = fundamentals.get("cashflow") or []
    n = min(len(inc), len(bs), len(cf))
    return [_derive_year(inc[i], bs[i], cf[i]) for i in range(n)]


# ============================================================================
# Ratio computation
# ============================================================================

def compute_metrics(symbol, api_key, discount_rate=0.10, growth_rate=0.08,
                    terminal_growth=0.03, projection_years=10,
                    growth_lookback=5):
    """
    Fetch + compute the full 80-indicator set for one ticker.
    Returns {'symbol', 'errors', 'values': {indicator_key: number_or_None}}.
    Values are raw numbers (percent indicators already ×100) — use
    format_value() for display.
    """
    fundamentals = fetch_fundamentals(symbol, api_key)
    years = derive_years(fundamentals)
    errors = list(fundamentals.get("errors", []))
    v = {}

    if not years:
        errors.append("No usable annual financial statements returned.")
        return {"symbol": fundamentals["symbol"], "errors": errors, "values": v}

    y0 = years[0]
    y1 = years[1] if len(years) > 1 else None
    quote = fundamentals.get("quote") or {}
    price = _num(quote.get("price"))
    market_cap = _num(quote.get("marketCap"))
    if market_cap is None and price is not None and y0.get("shares"):
        market_cap = price * y0["shares"]

    def yr(i):
        return years[i] if 0 <= i < len(years) else None

    def field_cagr(field, n):
        if not n:
            return None
        end_y, start_y = yr(0), yr(n)
        if not end_y or not start_y:
            return None
        return _cagr(end_y.get(field), start_y.get(field), n)

    # ── Enterprise Value ────────────────────────────────────────────────
    ev = None
    if market_cap is not None:
        ev = (market_cap + (y0.get("total_debt") or 0.0) + (y0.get("preferred_stock") or 0.0)
              + (y0.get("minority_interest") or 0.0) - (y0.get("cash") or 0.0))

    # ── Profitability (latest year, averaged balance-sheet bases) ────────
    avg_equity = _avg(y0.get("equity"), y1.get("equity") if y1 else None)
    avg_assets = _avg(y0.get("total_assets"), y1.get("total_assets") if y1 else None)
    avg_invested_cap = _avg(y0.get("invested_capital"), y1.get("invested_capital") if y1 else None)
    avg_capital_employed = _avg(y0.get("capital_employed"), y1.get("capital_employed") if y1 else None)

    v["roe"] = _pct(_div(y0.get("net_income"), avg_equity))
    v["roic"] = _pct(_div(y0.get("nopat"), avg_invested_cap))
    v["roce"] = _pct(_div(y0.get("ebit"), avg_capital_employed))
    v["roa"] = _pct(_div(y0.get("net_income"), avg_assets))
    v["gross_margin"] = _pct(_div(y0.get("gross_profit"), y0.get("revenue")))
    v["operating_margin"] = _pct(_div(y0.get("ebit"), y0.get("revenue")))
    v["ebitda_margin"] = _pct(_div(y0.get("ebitda"), y0.get("revenue")))
    v["net_margin"] = _pct(_div(y0.get("net_income"), y0.get("revenue")))
    v["fcf_margin"] = _pct(_div(y0.get("fcf"), y0.get("revenue")))
    v["owner_earnings_margin"] = _pct(_div(y0.get("owner_earnings"), y0.get("revenue")))

    # ── Growth ────────────────────────────────────────────────────────
    v["revenue_cagr_3y"] = _pct(field_cagr("revenue", 3))
    v["revenue_cagr_5y"] = _pct(field_cagr("revenue", 5))
    v["revenue_cagr_10y"] = _pct(field_cagr("revenue", 10))
    v["eps_cagr_3y"] = _pct(field_cagr("eps", 3))
    v["eps_cagr_5y"] = _pct(field_cagr("eps", 5))
    v["eps_cagr_10y"] = _pct(field_cagr("eps", 10))
    v["fcf_cagr_3y"] = _pct(field_cagr("fcf", 3))
    v["fcf_cagr_5y"] = _pct(field_cagr("fcf", 5))
    v["fcf_cagr_10y"] = _pct(field_cagr("fcf", 10))
    v["bvps_cagr"] = _pct(field_cagr("bvps", min(growth_lookback, len(years) - 1) or None))

    # ── Cash Flow ─────────────────────────────────────────────────────
    v["cfo"] = y0.get("cfo")
    v["fcf"] = y0.get("fcf")
    v["fcf_per_share"] = _div(y0.get("fcf"), y0.get("shares"))
    v["owner_earnings"] = y0.get("owner_earnings")
    v["owner_earnings_per_share"] = _div(y0.get("owner_earnings"), y0.get("shares"))
    v["fcf_over_ni"] = _pct(_div(y0.get("fcf"), y0.get("net_income")))
    v["cfo_over_ni"] = _pct(_div(y0.get("cfo"), y0.get("net_income")))
    v["capex_over_cfo"] = _pct(_div(y0.get("capex"), y0.get("cfo")))
    v["capex_over_revenue"] = _pct(_div(y0.get("capex"), y0.get("revenue")))
    v["owner_earnings_yield"] = _pct(_div(y0.get("owner_earnings"), market_cap))

    # ── Balance Sheet ─────────────────────────────────────────────────
    v["debt_to_equity"] = _div(y0.get("total_debt"), y0.get("equity"))
    v["net_debt_to_equity"] = _div(y0.get("net_debt"), y0.get("equity"))
    v["debt_to_ebitda"] = _div(y0.get("total_debt"), y0.get("ebitda"))
    v["net_debt_to_ebitda"] = _div(y0.get("net_debt"), y0.get("ebitda"))
    v["debt_to_fcf"] = _div(y0.get("total_debt"), y0.get("fcf"))
    v["net_debt_to_fcf"] = _div(y0.get("net_debt"), y0.get("fcf"))
    v["interest_coverage"] = _div(y0.get("ebit"), y0.get("interest_expense"))
    v["current_ratio"] = _div(y0.get("total_current_assets"), y0.get("total_current_liabilities"))
    quick_assets = None
    if y0.get("cash") is not None or y0.get("short_term_inv") is not None or y0.get("receivables") is not None:
        quick_assets = (y0.get("cash") or 0.0) + (y0.get("short_term_inv") or 0.0) + (y0.get("receivables") or 0.0)
    v["quick_ratio"] = _div(quick_assets, y0.get("total_current_liabilities"))
    v["cash_over_assets"] = _pct(_div(y0.get("cash"), y0.get("total_assets")))

    # ── Capital Efficiency ────────────────────────────────────────────
    avg_ar = _avg(y0.get("receivables"), y1.get("receivables") if y1 else None)
    avg_inv = _avg(y0.get("inventory"), y1.get("inventory") if y1 else None)
    avg_ap = _avg(y0.get("payables"), y1.get("payables") if y1 else None)
    avg_nwc = _avg(y0.get("operating_nwc"), y1.get("operating_nwc") if y1 else None)

    v["asset_turnover"] = _div(y0.get("revenue"), avg_assets)
    v["working_capital_turnover"] = _div(y0.get("revenue"), avg_nwc)
    v["inventory_turnover"] = _div(y0.get("cogs"), avg_inv)
    v["receivables_turnover"] = _div(y0.get("revenue"), avg_ar)
    v["payables_turnover"] = _div(y0.get("cogs"), avg_ap)
    v["dso"] = _div(avg_ar, y0.get("revenue"))
    v["dso"] = v["dso"] * 365 if v["dso"] is not None else None
    v["dio"] = _div(avg_inv, y0.get("cogs"))
    v["dio"] = v["dio"] * 365 if v["dio"] is not None else None
    v["dpo"] = _div(avg_ap, y0.get("cogs"))
    v["dpo"] = v["dpo"] * 365 if v["dpo"] is not None else None
    if v["dso"] is not None and v["dio"] is not None and v["dpo"] is not None:
        v["cash_conversion_cycle"] = v["dso"] + v["dio"] - v["dpo"]
    else:
        v["cash_conversion_cycle"] = None
    if y1 is not None:
        d_nopat = _num(y0.get("nopat")) - _num(y1.get("nopat")) \
            if y0.get("nopat") is not None and y1.get("nopat") is not None else None
        d_ic = _num(y0.get("invested_capital")) - _num(y1.get("invested_capital")) \
            if y0.get("invested_capital") is not None and y1.get("invested_capital") is not None else None
        v["incremental_roic"] = _pct(_div(d_nopat, d_ic)) if d_ic not in (None, 0) else None
    else:
        v["incremental_roic"] = None

    # ── Shareholder Economics ─────────────────────────────────────────
    n_hist = min(growth_lookback, len(years) - 1) or None
    v["shares_cagr"] = _pct(field_cagr("shares", n_hist))
    v["eps_growth"] = _pct(_cagr(y0.get("eps"), y1.get("eps"), 1)) if y1 else None
    v["fcf_per_share_growth"] = None
    if y1 is not None:
        fcf_ps_0 = _div(y0.get("fcf"), y0.get("shares"))
        fcf_ps_1 = _div(y1.get("fcf"), y1.get("shares"))
        v["fcf_per_share_growth"] = _pct(_cagr(fcf_ps_0, fcf_ps_1, 1))
    div_per_share = _div(y0.get("dividends_paid"), y0.get("shares"))
    v["dividend_yield"] = _pct(_div(div_per_share, price))
    v["dividend_cagr"] = _pct(field_cagr("dividends_paid", n_hist))
    v["dividend_payout"] = _pct(_div(y0.get("dividends_paid"), y0.get("net_income")))
    v["fcf_payout"] = _pct(_div(y0.get("dividends_paid"), y0.get("fcf")))
    v["buyback_yield"] = _pct(_div(y0.get("net_buybacks"), market_cap))
    v["net_dilution"] = _pct(_cagr(y0.get("shares"), y1.get("shares"), 1)) if y1 else None
    v["retained_earnings_growth"] = _pct(field_cagr("retained_earnings", n_hist))

    # ── Valuation ─────────────────────────────────────────────────────
    v["pe"] = _div(price, y0.get("eps"))
    v["forward_pe"] = _div(price, fundamentals.get("forward_eps"))
    v["p_fcf"] = _div(market_cap, y0.get("fcf"))
    v["p_s"] = _div(market_cap, y0.get("revenue"))
    v["p_b"] = _div(market_cap, y0.get("equity"))
    v["ev_sales"] = _div(ev, y0.get("revenue"))
    v["ev_ebit"] = _div(ev, y0.get("ebit"))
    v["ev_ebitda"] = _div(ev, y0.get("ebitda"))
    v["ev_fcf"] = _div(ev, y0.get("fcf"))
    v["earnings_yield"] = _pct(_div(y0.get("eps"), price))
    v["fcf_yield"] = _pct(_div(y0.get("fcf"), market_cap))
    v["owner_earnings_yield_val"] = v["owner_earnings_yield"]

    # ── Economic Valuation / intrinsic value ─────────────────────────
    lookback_years = years[:growth_lookback + 1] if len(years) > 1 else years
    ni_hist = [y.get("net_income") for y in lookback_years if y.get("net_income") is not None]
    fcf_hist = [y.get("fcf") for y in lookback_years if y.get("fcf") is not None]
    v["normalized_earnings"] = float(np.mean(ni_hist)) if ni_hist else None
    v["normalized_fcf"] = float(np.mean(fcf_hist)) if fcf_hist else None
    v["owner_earnings_ev"] = y0.get("owner_earnings")

    oe0 = y0.get("owner_earnings")
    shares0 = y0.get("shares")
    intrinsic_value = None
    if oe0 is not None and oe0 > 0 and discount_rate > terminal_growth:
        pv_sum = 0.0
        oe_t = oe0
        for t in range(1, int(projection_years) + 1):
            oe_t = oe_t * (1 + growth_rate)
            pv_sum += oe_t / ((1 + discount_rate) ** t)
        terminal_value = oe_t * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** int(projection_years))
        intrinsic_value = pv_sum + pv_terminal
    v["intrinsic_value"] = intrinsic_value
    iv_per_share = _div(intrinsic_value, shares0)
    v["intrinsic_value_per_share"] = iv_per_share
    v["price_to_intrinsic"] = _div(price, iv_per_share)
    v["margin_of_safety"] = _pct(1 - v["price_to_intrinsic"]) if v["price_to_intrinsic"] is not None else None

    oe_growth = field_cagr("owner_earnings", n_hist)
    net_buyback_yield = _div(y0.get("net_buybacks"), market_cap)
    # Expected Long-Term Shareholder Return ≈ Owner Earnings Yield + Owner
    # Earnings Growth + Net Buyback Yield ± Valuation Multiple Change. The
    # multiple-change term is inherently a forward-looking assumption about
    # re-rating, not something derivable from the statements, so it's
    # treated as 0 here (a "no re-rating" baseline) rather than guessed at.
    expected_return = None
    oey = _div(oe0, market_cap)
    if oey is not None:
        expected_return = oey + (oe_growth or 0.0) + (net_buyback_yield or 0.0)
    v["expected_return"] = _pct(expected_return)

    return {"symbol": fundamentals["symbol"], "errors": errors, "values": v}
