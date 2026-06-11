"""
CANSLIM Multi-Ticker Scoring & Ranking Page
Scores any number of companies on the CANSLIM fundamentals methodology
and presents a ranked table with per-ticker drill-down.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # find canslim_module

import streamlit as st
import pandas as pd
from canslim_module import score_canslim_universe

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

def pct(val, decimals=1):
    """Format a decimal as a percentage string, or '—' if None."""
    if val is None:
        return "—"
    return f"{val * 100:.{decimals}f}%"


def fmt_val(criterion_key, val):
    """Format a metric value for display in the detailed metrics table."""
    if val is None:
        return "no data"
    if criterion_key == 'eps_accel':
        return "✅ TRUE" if val else "❌ FALSE"
    if criterion_key == 'inst_pct':
        return pct(val)
    return pct(val)


def traffic_light(met):
    if met is True:
        return "✅"
    if met is False:
        return "❌"
    return "⚠️ no data"


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("⚙️ CANSLIM Configuration")

st.sidebar.subheader("🎯 Tickers")
ticker_input = st.sidebar.text_area(
    "Enter tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL, NVDA, AMZN",
    height=100,
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

run_btn = st.sidebar.button("🚀 Run CANSLIM Analysis", type="primary", use_container_width=True)

# ============================================================================
# MAIN — RUN ANALYSIS
# ============================================================================

if run_btn:
    symbols = [s.strip().upper() for s in ticker_input.replace("\n", ",").split(",") if s.strip()]
    if not symbols:
        st.warning("Enter at least one ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching fundamental data for {len(symbols)} ticker(s)…"):
        results = score_canslim_universe(symbols)

    # ── Ranked summary table ─────────────────────────────────────────────
    st.subheader("🏆 Ranked Scoring Table")

    rows = []
    rank = 1
    prev_score = None
    for i, r in enumerate(results):
        if r['score'] != prev_score:
            rank = i + 1
        rows.append({
            'Rank':               rank,
            'Ticker':             r['symbol'],
            'Total Score (/100)': r['score'],
            'Criteria Met':       f"{r['criteria_met']} / 10",
            'Data Gaps':          r['data_gaps'],
        })
        prev_score = r['score']

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Per-ticker detailed breakdown ────────────────────────────────────
    st.subheader("🔍 Detailed Breakdown")
    st.caption("Expand any ticker to see its full metrics, scoring, and quarterly trend.")

    for r in results:
        sym = r['symbol']
        score = r['score']
        gaps  = r['data_gaps']

        with st.expander(f"{sym}  —  {score}/100   |  {r['criteria_met']}/10 criteria met  |  {gaps} data gap(s)"):

            # ── 1. Metrics table ─────────────────────────────────────────
            st.markdown("**Computed Metrics**")
            m = r['metrics']
            latest_q = r['q_dates'][0] if r['q_dates'] else "—"
            latest_a = r['a_dates'][0] if r['a_dates'] else "—"

            metrics_rows = [
                ("#1–4",  "EPS YoY Growth Q4 / Q3 / Q2 / Q1",
                 f"{pct(m.get('eps_gr_q4'))} / {pct(m.get('eps_gr_q3'))} / {pct(m.get('eps_gr_q2'))} / {pct(m.get('eps_gr_q1'))}",
                 latest_q),
                ("#3",    "EPS Acceleration (3 Qtrs)",
                 fmt_val('eps_accel', m.get('eps_accel')), latest_q),
                ("#4",    "EPS 3Y Avg Annual Growth",
                 pct(m.get('eps_3y_avg')), latest_a),
                ("#5",    "EPS 5Y Avg Annual Growth",
                 pct(m.get('eps_5y_avg')), latest_a),
                ("#6",    "Revenue YoY Growth Q4",
                 pct(m.get('rev_gr_q4')), latest_q),
                ("#7",    "Avg Revenue Growth (Q2–Q4)",
                 pct(m.get('rev_3q_avg')), latest_q),
                ("#8",    "Pretax Income Annual Growth",
                 pct(m.get('pretax_ann_gr')), latest_a),
                ("#9",    "Net Income Annual Growth",
                 pct(m.get('ni_ann_gr')), latest_a),
                ("#10a",  "ROE Q4 (latest)",
                 pct(m.get('roe_q4')), latest_q),
                ("#10b",  "ROE YoY Growth",
                 pct(m.get('roe_yoy_gr')), latest_q),
                ("#11",   "Institutional Ownership",
                 pct(m.get('inst_pct')), "latest"),
            ]
            st.dataframe(
                pd.DataFrame(metrics_rows, columns=["#", "Metric", "Value", "Period"]),
                use_container_width=True, hide_index=True
            )

            # ── 2. Scoring table ─────────────────────────────────────────
            st.markdown("**Scoring Criteria**")
            score_rows = []
            for sd in r['score_details']:
                key = next((k for k, v in m.items() if v == sd['Value']), sd['Criterion'])
                score_rows.append({
                    'Criterion':  sd['Criterion'],
                    'Value':      pct(sd['Value']) if sd['Value'] is not None and sd['Criterion'] != 'EPS Acceleration (3 Qtrs)' else fmt_val('eps_accel', sd['Value']),
                    'Threshold':  sd['Threshold'],
                    'Points':     sd['Points'],
                    'Status':     traffic_light(sd['Met']),
                })
            st.dataframe(
                pd.DataFrame(score_rows),
                use_container_width=True, hide_index=True
            )

            # ── 3. Total score ────────────────────────────────────────────
            st.metric(f"Total CANSLIM Score: {sym}", f"{score} / 100")

            # ── 4. Quarterly EPS & Revenue growth trend ──────────────────
            st.markdown("**Last 4 Quarters — YoY Growth Trend**")
            q_labels = ["Q1 (oldest)", "Q2", "Q3", "Q4 (latest)"]
            trend_rows = []
            for i, lbl in enumerate(q_labels):
                eg = r['eps_gr_series'][i] if i < len(r['eps_gr_series']) else None
                rg = r['rev_gr_series'][i] if i < len(r['rev_gr_series']) else None
                trend_rows.append({'Quarter': lbl, 'EPS YoY': pct(eg), 'Revenue YoY': pct(rg)})
            st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

            # ── 5. Data gaps / errors ─────────────────────────────────────
            if r['errors']:
                st.caption("⚠️ Data gaps: " + " · ".join(r['errors']))

    st.info("Select a ticker above to expand its full CANSLIM breakdown.")

else:
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
