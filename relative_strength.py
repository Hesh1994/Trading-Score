"""
Turnover-based Relative Strength (RS).

RS(t) = Stock Trading Value(t) / Benchmark Total Trading Value(t)

where Trading Value = Close x Volume. The benchmark trading value is either
its officially reported turnover, the sum of Close x Volume across all its
constituents, or an ETF proxy (e.g. SPY for the S&P 500).

Pure-pandas, no TA-lib, matching the style of scoring_module.py.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ------------------------------------------------------ exchange index proxy

# Exchange code (as used by canslim_module.COUNTRY_EXCHANGES) → a liquid,
# broad ETF whose Close x Volume approximates that exchange's major-index
# turnover. Used to auto-suggest a benchmark for "the exchange I chose in
# the scoring dashboard." Not exhaustive — exchanges with no reasonably
# liquid single-country ETF are simply absent, and the RS page falls back to
# other benchmark modes for those.
EXCHANGE_INDEX_PROXY = {
    # Americas
    "NASDAQ": ("QQQ", "Nasdaq 100"),
    "NYSE":   ("SPY", "S&P 500"),
    "AMEX":   ("SPY", "S&P 500"),
    "TSX":    ("EWC", "iShares MSCI Canada"),
    "TSXV":   ("EWC", "iShares MSCI Canada"),
    "BVMF":   ("EWZ", "iShares MSCI Brazil"),
    "BMV":    ("EWW", "iShares MSCI Mexico"),
    "BCS":    ("ECH", "iShares MSCI Chile"),
    # Europe
    "LSE":      ("EWU", "iShares MSCI United Kingdom"),
    "ETR":      ("EWG", "iShares MSCI Germany"),
    "FRA":      ("EWG", "iShares MSCI Germany"),
    "EURONEXT": ("EWQ", "iShares MSCI France"),
    "AMS":      ("EWN", "iShares MSCI Netherlands"),
    "MIL":      ("EWI", "iShares MSCI Italy"),
    "MCE":      ("EWP", "iShares MSCI Spain"),
    "SIX":      ("EWL", "iShares MSCI Switzerland"),
    "STO":      ("EWD", "iShares MSCI Sweden"),
    "OSL":      ("NORW", "Global X MSCI Norway"),
    "WSE":      ("EPOL", "iShares MSCI Poland"),
    "IST":      ("TUR", "iShares MSCI Turkey"),
    # Asia Pacific
    "TYO":  ("EWJ", "iShares MSCI Japan"),
    "SHH":  ("ASHR", "Xtrackers CSI 300 China A"),
    "SHZ":  ("ASHR", "Xtrackers CSI 300 China A"),
    "HKSE": ("EWH", "iShares MSCI Hong Kong"),
    "KSC":  ("EWY", "iShares MSCI South Korea"),
    "NSE":  ("INDA", "iShares MSCI India"),
    "BSE":  ("INDA", "iShares MSCI India"),
    "ASX":  ("EWA", "iShares MSCI Australia"),
    "SES":  ("EWS", "iShares MSCI Singapore"),
    "KLSE": ("EWM", "iShares MSCI Malaysia"),
    "SET":  ("THD", "iShares MSCI Thailand"),
    "IDX":  ("EIDO", "iShares MSCI Indonesia"),
    "PSE":  ("EPHE", "iShares MSCI Philippines"),
    "TAI":  ("EWT", "iShares MSCI Taiwan"),
    "HOSE": ("VNM", "VanEck Vietnam"),
    # Middle East
    "SAU": ("KSA", "iShares MSCI Saudi Arabia"),
    "ADX": ("UAE", "iShares MSCI UAE"),
    "DFM": ("UAE", "iShares MSCI UAE"),
    "QSE": ("QAT", "iShares MSCI Qatar"),
    "TASE": ("EIS", "iShares MSCI Israel"),
    # Africa
    "EGX": ("EGPT", "VanEck Egypt Index"),
    "JSE": ("EZA", "iShares MSCI South Africa"),
}


def exchange_index_proxy(exchange_codes):
    """
    Best matching (etf, display_name) for a list of exchange codes, or None
    if none of them have a known proxy. Prefers the first code that matches.
    """
    for code in exchange_codes or []:
        if code in EXCHANGE_INDEX_PROXY:
            return EXCHANGE_INDEX_PROXY[code]
    return None


# ---------------------------------------------------------------- core math

def trading_value(df, price_col='Close', volume_col='Volume'):
    """Close x Volume per bar. Returns a Series indexed like df."""
    return (df[price_col].astype(float) * df[volume_col].astype(float)).rename('TradingValue')


def calculate_rs(stock_value, benchmark_value):
    """
    RS(t) = stock trading value / benchmark trading value, aligned on the
    intersection of both indexes. Zero/missing benchmark values become NaN
    rather than inf.
    """
    stock_value = pd.Series(stock_value).astype(float)
    benchmark_value = pd.Series(benchmark_value).astype(float)

    idx = stock_value.index.intersection(benchmark_value.index)
    s = stock_value.reindex(idx)
    b = benchmark_value.reindex(idx).replace(0, np.nan)

    return (s / b).rename('RS')


def average_rs(rs, interval=None):
    """
    Average RS over the last `interval` periods (all periods if interval is None).
    NaNs are excluded from both the sum and the count.
    """
    rs = pd.Series(rs).dropna()
    if interval is not None:
        rs = rs.tail(int(interval))
    return float(rs.mean()) if len(rs) else float('nan')


def rolling_average_rs(rs, interval):
    """Rolling mean of RS over `interval` periods."""
    return pd.Series(rs).rolling(int(interval), min_periods=1).mean().rename('AvgRS_%d' % int(interval))


def rs_trend(rs, interval=None):
    """
    Direction of RS over the interval, via ordinary least squares on the
    period index. Returns a dict with slope, slope as a percent of mean RS per
    period, first/last/mean RS, and an Up/Down/Flat label.

    Flat when the slope per period is under 0.5% of the mean RS.
    """
    rs = pd.Series(rs).dropna()
    if interval is not None:
        rs = rs.tail(int(interval))

    if len(rs) < 2:
        return {'slope': float('nan'), 'slope_pct_per_period': float('nan'),
                'first': float('nan'), 'last': float('nan'),
                'mean': float('nan'), 'direction': 'Insufficient data',
                'periods': len(rs)}

    x = np.arange(len(rs), dtype=float)
    slope, _ = np.polyfit(x, rs.values, 1)
    mean = float(rs.mean())
    slope_pct = (slope / mean * 100.0) if mean else float('nan')

    if not np.isfinite(slope_pct) or abs(slope_pct) < 0.5:
        direction = 'Flat'
    else:
        direction = 'Up' if slope > 0 else 'Down'

    return {'slope': float(slope), 'slope_pct_per_period': float(slope_pct),
            'first': float(rs.iloc[0]), 'last': float(rs.iloc[-1]),
            'mean': mean, 'direction': direction, 'periods': len(rs)}


def normalize_by_market_cap(rs, stock_market_cap, benchmark_market_cap):
    """
    Cap-weight-adjusted RS: RS(t) / (stock cap / benchmark cap).

    A value of 1.0 means the stock trades exactly in line with its index
    weight; above 1.0 means participation exceeds its weight. Scalars or
    Series are both accepted for the cap arguments.
    """
    rs = pd.Series(rs).astype(float)

    if np.isscalar(stock_market_cap):
        cap = pd.Series(float(stock_market_cap), index=rs.index)
    else:
        cap = pd.Series(stock_market_cap).reindex(rs.index).astype(float)

    if np.isscalar(benchmark_market_cap):
        bench = pd.Series(float(benchmark_market_cap), index=rs.index)
    else:
        bench = pd.Series(benchmark_market_cap).reindex(rs.index).astype(float)

    weight = cap / bench.replace(0, np.nan)
    return (rs / weight.replace(0, np.nan)).rename('RS_CapAdjusted')


# ------------------------------------------------------------- benchmark TV

def benchmark_trading_value_from_constituents(constituent_data, price_col='Close',
                                              volume_col='Volume', min_coverage=0.0):
    """
    Sum Close x Volume across all constituents.

    constituent_data: dict {ticker: DataFrame}, or a long DataFrame with a
    'Ticker' column. Days where fewer than `min_coverage` (0-1) of the
    constituents reported are set to NaN, so a partial session cannot inflate RS.
    """
    if isinstance(constituent_data, pd.DataFrame):
        if 'Ticker' not in constituent_data.columns:
            raise ValueError("long-format benchmark data needs a 'Ticker' column")
        constituent_data = {t: g for t, g in constituent_data.groupby('Ticker')}

    values = {}
    for ticker, df in constituent_data.items():
        if df is None or df.empty:
            continue
        if price_col not in df.columns or volume_col not in df.columns:
            continue
        values[ticker] = trading_value(df, price_col, volume_col)

    if not values:
        raise ValueError('no usable constituent data')

    matrix = pd.DataFrame(values)
    total = matrix.sum(axis=1, skipna=True)

    if min_coverage > 0:
        reported = matrix.notna().sum(axis=1) / float(matrix.shape[1])
        total[reported < min_coverage] = np.nan

    return total.rename('BenchmarkTradingValue')


def benchmark_trading_value_from_proxy(df, price_col='Close', volume_col='Volume'):
    """Benchmark turnover approximated by an ETF proxy (e.g. SPY, QQQ)."""
    return trading_value(df, price_col, volume_col).rename('BenchmarkTradingValue')


def turnover_series_from_frame(df, date_col=None, value_col=None):
    """
    Coerce a table of officially reported turnover into a date-indexed Series.

    Column names are auto-detected when not given: the date column is the first
    whose name contains 'date' (else the first column), and the value column is
    the first remaining one naming turnover/value/volume (else the next column).
    Unparseable dates are dropped and the result is sorted.
    """
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError('need at least a date column and a turnover column')

    if date_col is None:
        date_col = next((c for c in cols if 'date' in str(c).lower()), cols[0])
    if value_col is None:
        value_col = next(
            (c for c in cols
             if c != date_col and any(k in str(c).lower()
                                      for k in ('turnover', 'value', 'volume'))),
            next(c for c in cols if c != date_col))

    out = pd.Series(pd.to_numeric(df[value_col], errors='coerce').values,
                    index=pd.to_datetime(df[date_col], errors='coerce'),
                    name='BenchmarkTradingValue')
    return out[out.index.notna()].sort_index()

# ----------------------------------------------------------------- reporting

def build_rs_table(stock_df, benchmark_value, benchmark_name='Benchmark',
                   interval=None, price_col='Close', volume_col='Volume',
                   stock_market_cap=None, benchmark_market_cap=None):
    """
    Full per-period table plus an 'Average RS' summary row.

    Columns: Date, Stock Trading Value, <benchmark> Trading Value, RS
    (and RS (Cap-Adjusted) when market cap data is supplied).
    Returns (table, summary_dict).
    """
    stock_value = trading_value(stock_df, price_col, volume_col)
    bench_value = pd.Series(benchmark_value).astype(float)

    rs = calculate_rs(stock_value, bench_value)
    if interval is not None:
        rs = rs.tail(int(interval))

    bench_col = '%s Trading Value' % benchmark_name
    table = pd.DataFrame({
        'Date': rs.index,
        'Stock Trading Value': stock_value.reindex(rs.index).values,
        bench_col: bench_value.reindex(rs.index).values,
        'RS': rs.values,
    })

    cap_adjusted = None
    if stock_market_cap is not None and benchmark_market_cap is not None:
        cap_adjusted = normalize_by_market_cap(rs, stock_market_cap, benchmark_market_cap)
        table['RS (Cap-Adjusted)'] = cap_adjusted.values

    avg = average_rs(rs)
    trend = rs_trend(rs)

    summary_row = {
        'Date': 'Average RS',
        'Stock Trading Value': table['Stock Trading Value'].mean(),
        bench_col: table[bench_col].mean(),
        'RS': avg,
    }
    if cap_adjusted is not None:
        summary_row['RS (Cap-Adjusted)'] = average_rs(cap_adjusted)

    table = pd.concat([table, pd.DataFrame([summary_row])], ignore_index=True)

    summary = {
        'average_rs': avg,
        'average_rs_cap_adjusted': average_rs(cap_adjusted) if cap_adjusted is not None else None,
        'benchmark': benchmark_name,
        'interval': int(interval) if interval else int(len(rs)),
    }
    summary.update(trend)
    return table, summary


def format_rs_table(table, float_fmt='{:,.0f}', rs_fmt='{:.6f}'):
    """Table with trading values and RS rendered as readable strings."""
    out = table.copy()
    for col in out.columns:
        if 'Trading Value' in col:
            out[col] = out[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else '-')
        elif col.startswith('RS'):
            out[col] = out[col].map(lambda v: rs_fmt.format(v) if pd.notna(v) else '-')
    return out


# ----------------------------------------------------------------- yfinance

def fetch_ohlcv(tickers, start, end, interval='1d', auto_adjust=False):
    """
    Download OHLCV via yfinance. Returns {ticker: DataFrame}.
    auto_adjust=False keeps raw closes, so Close x Volume is actual turnover.
    """
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]

    raw = yf.download(tickers, start=start, end=end, interval=interval,
                      auto_adjust=auto_adjust, progress=False,
                      group_by='ticker', threads=True)

    data = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker in raw.columns.get_level_values(0):
                df = raw[ticker].dropna(how='all')
                if not df.empty:
                    data[ticker] = df
    elif not raw.empty:
        data[tickers[0]] = raw.dropna(how='all')

    return data


def compute_rs_from_yfinance(ticker, start, end, benchmark_proxy=None,
                             benchmark_constituents=None, benchmark_name=None,
                             interval=None, bar_interval='1d',
                             stock_market_cap=None, benchmark_market_cap=None):
    """
    End-to-end: download, build benchmark turnover, return (table, summary).

    Pass exactly one of benchmark_proxy (an ETF ticker) or
    benchmark_constituents (a list of index members).
    """
    if (benchmark_proxy is None) == (benchmark_constituents is None):
        raise ValueError('pass exactly one of benchmark_proxy or benchmark_constituents')

    stock_data = fetch_ohlcv(ticker, start, end, bar_interval)
    if ticker not in stock_data:
        raise ValueError('no data returned for %s' % ticker)

    if benchmark_proxy is not None:
        proxy_data = fetch_ohlcv(benchmark_proxy, start, end, bar_interval)
        if benchmark_proxy not in proxy_data:
            raise ValueError('no data returned for benchmark %s' % benchmark_proxy)
        bench_value = benchmark_trading_value_from_proxy(proxy_data[benchmark_proxy])
        name = benchmark_name or benchmark_proxy
    else:
        members = fetch_ohlcv(list(benchmark_constituents), start, end, bar_interval)
        bench_value = benchmark_trading_value_from_constituents(members, min_coverage=0.5)
        name = benchmark_name or 'Benchmark'

    return build_rs_table(stock_data[ticker], bench_value, benchmark_name=name,
                          interval=interval, stock_market_cap=stock_market_cap,
                          benchmark_market_cap=benchmark_market_cap)


# --------------------------------------------------------------------- CLI

def _main():
    import argparse
    import os

    p = argparse.ArgumentParser(description='Turnover-based Relative Strength')
    p.add_argument('ticker')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--benchmark-proxy', help='ETF proxy for the index, e.g. SPY')
    p.add_argument('--benchmark-constituents',
                   help='comma-separated members, or a path to a file with one per line')
    p.add_argument('--benchmark-name')
    p.add_argument('--interval', type=int, help='periods to report, e.g. 14')
    p.add_argument('--bar-interval', default='1d', help='1d, 1wk, 1mo')
    p.add_argument('--stock-market-cap', type=float)
    p.add_argument('--benchmark-market-cap', type=float)
    p.add_argument('--csv', help='write the table to this path')
    args = p.parse_args()

    members = None
    if args.benchmark_constituents:
        if os.path.exists(args.benchmark_constituents):
            with open(args.benchmark_constituents) as fh:
                members = [ln.strip() for ln in fh if ln.strip()]
        else:
            members = [t.strip() for t in args.benchmark_constituents.split(',') if t.strip()]

    table, summary = compute_rs_from_yfinance(
        args.ticker, args.start, args.end,
        benchmark_proxy=args.benchmark_proxy,
        benchmark_constituents=members,
        benchmark_name=args.benchmark_name,
        interval=args.interval,
        bar_interval=args.bar_interval,
        stock_market_cap=args.stock_market_cap,
        benchmark_market_cap=args.benchmark_market_cap,
    )

    print(format_rs_table(table).to_string(index=False))
    print()
    print('Average RS (%d periods vs %s): %.6f'
          % (summary['interval'], summary['benchmark'], summary['average_rs']))
    if summary['average_rs_cap_adjusted'] is not None:
        print('Average RS, cap-adjusted: %.6f' % summary['average_rs_cap_adjusted'])
    print('Trend: %s (%+.2f%% of mean per period, %.6f -> %.6f)'
          % (summary['direction'], summary['slope_pct_per_period'],
             summary['first'], summary['last']))

    if args.csv:
        table.to_csv(args.csv, index=False)
        print('Wrote %s' % args.csv)


if __name__ == '__main__':
    _main()
