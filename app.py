import io
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF


# -------------------- Page config --------------------
st.set_page_config(page_title="SF Business Registrations Dashboard", layout="wide")
st.title("San Francisco Business Registrations Dashboard")
st.caption("Real-time insights from DataSF · Registered Business Locations (g8m3-pdis)")


# -------------------- Secrets / Socrata config --------------------
SOC_DOMAIN = st.secrets.get("socrata", {}).get("domain", "data.sfgov.org")
DATASET_ID = st.secrets.get("socrata", {}).get("dataset_id", "g8m3-pdis")
APP_TOKEN  = st.secrets.get("socrata", {}).get("app_token", None)
USERNAME   = st.secrets.get("socrata", {}).get("username", None)  # optional
PASSWORD   = st.secrets.get("socrata", {}).get("password", None)  # optional

BASE_URL = f"https://{SOC_DOMAIN}/resource/{DATASET_ID}.json"

HEADERS = {}
if APP_TOKEN:
    HEADERS["X-App-Token"] = APP_TOKEN

AUTH = (USERNAME, PASSWORD) if (USERNAME and PASSWORD) else None


# -------------------- Helper utilities --------------------
def _dt_iso(d: date) -> str:
    """ISO 8601 midnight timestamp string for Socrata."""
    return f"{d.isoformat()}T00:00:00.000"


def _with_app_token(params: dict) -> dict:
    """Also pass $$app_token in query string for broader compatibility."""
    if APP_TOKEN and "$$app_token" not in params:
        params = {**params, "$$app_token": APP_TOKEN}
    return params


def socrata_get(params: dict, timeout=45, max_retries=3):
    """
    GET wrapper with retries, rate-limit backoff and human-readable diagnostics.
    Raises on failure after retries.
    """
    params = _with_app_token(params)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                BASE_URL, params=params, headers=HEADERS, auth=AUTH, timeout=timeout
            )
            # Handle rate limiting
            if resp.status_code == 429:
                wait = 2 ** attempt
                st.warning(f"Socrata rate limited (HTTP 429). Retrying in {wait}s …")
                time.sleep(wait)
                continue

            # Non-OK -> show snippet and raise
            if not resp.ok:
                snippet = (resp.text or "")[:300].replace("\n", " ")
                st.error(f"Socrata HTTP {resp.status_code} · snippet: {snippet}")
                resp.raise_for_status()

            return resp
        except requests.RequestException as e:
            last_err = e
            # small linear/exponential backoff
            time.sleep(1.5 * attempt)

    # After retries, bubble up the last error
    raise last_err


def ping_socrata():
    """Very small request to expose 401/403/429/400 early."""
    try:
        r = socrata_get({"$select": "count(ttxid)", "$limit": 1}, timeout=20)
        _ = r.json()
        st.sidebar.success("Socrata connectivity: OK")
    except Exception as e:
        st.sidebar.error("Socrata connectivity failed. Check domain/token/limits.")
        st.sidebar.exception(e)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_range(start_d: date, end_exclusive_d: date) -> pd.DataFrame:
    """
    Fetch all rows whose location_start_date in [start_d, end_exclusive_d).
    Handles pagination (1000 rows per page). Returns a normalized DataFrame.
    """
    where = (
        f"location_start_date >= '{_dt_iso(start_d)}' AND "
        f"location_start_date < '{_dt_iso(end_exclusive_d)}'"
    )

    select_cols = [
        "ttxid",
        "location_start_date",
        "naic_code_description",
        "neighborhoods_analysis_boundaries",
    ]

    # 1) Count first (fast)
    params_count = {"$select": "count(ttxid)", "$where": where}
    r = socrata_get(params_count, timeout=30)
    j = r.json()
    total = int(j[0].get("count_ttxid", 0)) if j else 0
    if total == 0:
        return pd.DataFrame(columns=select_cols + ["start_date"])

    # 2) Page through
    all_rows = []
    limit = 1000
    for offset in range(0, total, limit):
        params = {
            "$select": ", ".join(select_cols),
            "$where": where,
            "$order": "location_start_date",
            "$limit": min(limit, total - offset),
            "$offset": offset,
        }
        rr = socrata_get(params, timeout=45)
        all_rows.extend(rr.json())

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Normalize fields
    if "naic_code_description" not in df.columns:
        df["naic_code_description"] = np.nan
    if "neighborhoods_analysis_boundaries" not in df.columns:
        df["neighborhoods_analysis_boundaries"] = np.nan

    df["naic_code_description"] = df["naic_code_description"].fillna("Unknown")
    df["neighborhoods_analysis_boundaries"] = df["neighborhoods_analysis_boundaries"].fillna(
        "Outside San Francisco"
    )

    # Parse date only
    df["start_date"] = pd.to_datetime(df["location_start_date"], errors="coerce").dt.date
    return df


def counts_block(df: pd.DataFrame, today: date, sow: date, eow: date, som: date, eom: date):
    if df.empty:
        return 0, 0, 0
    today_c = int((df["start_date"] == today).sum())
    week_c = int(((df["start_date"] >= sow) & (df["start_date"] < eow)).sum())
    month_c = int(((df["start_date"] >= som) & (df["start_date"] < eom)).sum())
    return today_c, week_c, month_c


def render_hbar(counts: pd.Series, title: str):
    # Per instruction: use matplotlib, single plot, do not set specific colors
    fig = plt.figure(figsize=(8, max(4, 0.3 * len(counts))))
    ax = fig.add_subplot(1, 1, 1)
    y = np.arange(len(counts))
    ax.barh(y, counts.values)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of New Businesses")
    ax.set_title(title)
    st.pyplot(fig)
    return fig


def make_pdf(period_label: str, today_c: int, week_c: int, month_c: int,
             fig1_path: str, fig2_path: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "SF Business Registrations Report", ln=1, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Period: {period_label}", ln=1)
    pdf.ln(2)
    pdf.cell(0, 8, f"New Businesses Today: {today_c}", ln=1)
    pdf.cell(0, 8, f"New Businesses This Week: {week_c}", ln=1)
    pdf.cell(0, 8, f"New Businesses This Month: {month_c}", ln=1)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "By Industry", ln=1)
    pdf.image(fig1_path, x=10, w=185)
    pdf.ln(3)
    pdf.cell(0, 8, "By Neighborhood", ln=1)
    pdf.image(fig2_path, x=10, w=185)
    return pdf.output(dest="S").encode("latin-1")


# -------------------- Sidebar diagnostics --------------------
with st.sidebar:
    st.markdown("### Diagnostics")
    st.write(f"Domain: `{SOC_DOMAIN}` · Dataset: `{DATASET_ID}`")
    st.write(f"App token set: {'Yes' if APP_TOKEN else 'No'}")
    ping_socrata()


# -------------------- Date ranges (today / week / month) --------------------
today = date.today()

# Week: Monday as start
start_of_week = today - timedelta(days=today.weekday())
start_of_next_week = start_of_week + timedelta(days=7)

# Month
start_of_month = today.replace(day=1)
start_of_next_month = (
    start_of_month.replace(month=start_of_month.month % 12 + 1, day=1)
    if start_of_month.month < 12
    else start_of_month.replace(year=start_of_month.year + 1, month=1, day=1)
)

# Fetch superset covering this week & this month once (minimizes API calls)
earliest_needed = min(start_of_week, start_of_month)
latest_needed = max(start_of_next_week, start_of_next_month)

with st.spinner("Fetching live data from Socrata..."):
    superset_df = fetch_range(earliest_needed, latest_needed)

# KPIs
today_c, week_c, month_c = counts_block(
    superset_df, today, start_of_week, start_of_next_week, start_of_month, start_of_next_month
)
k1, k2, k3 = st.columns(3)
k1.metric("New Businesses Today", today_c)
k2.metric("New This Week", week_c)
k3.metric("New This Month", month_c)

st.markdown("### Explore by Period, Industry, and Neighborhood")
period_choice = st.radio("Time period", ["Today", "This Week", "This Month", "Custom"], index=0, horizontal=True)

# Resolve selected range & dataframe
if period_choice == "Today":
    df_period = superset_df[superset_df["start_date"] == today]
    period_label = today.strftime("%b %d, %Y")
elif period_choice == "This Week":
    mask = (superset_df["start_date"] >= start_of_week) & (superset_df["start_date"] < start_of_next_week)
    df_period = superset_df[mask]
    period_label = f"{start_of_week.strftime('%b %d')} – {(start_of_next_week - timedelta(days=1)).strftime('%b %d, %Y')}"
elif period_choice == "This Month":
    mask = (superset_df["start_date"] >= start_of_month) & (superset_df["start_date"] < start_of_next_month)
    df_period = superset_df[mask]
    period_label = start_of_month.strftime("%B %Y")
else:
    default_start = today - timedelta(days=7)
    default_end = today
    start_d, end_d = st.date_input("Select date range", value=(default_start, default_end))
    if isinstance(start_d, tuple):  # compatibility for older Streamlit builds
        start_d, end_d = start_d
    end_exclusive = end_d + timedelta(days=1)
    with st.spinner("Fetching custom range..."):
        df_period = fetch_range(start_d, end_exclusive)
    period_label = f"{start_d.strftime('%b %d, %Y')} – {end_d.strftime('%b %d, %Y')}"

if df_period.empty:
    st.info("No business registrations found for the selected period.")
    st.stop()

# -------------------- Charts --------------------
st.markdown("#### New Businesses by Industry")
industry_counts = df_period["naic_code_description"].value_counts(ascending=True)
fig_industry = render_hbar(industry_counts, f"New Businesses by Industry ({period_label})")

st.markdown("#### New Businesses by Neighborhood")
neigh_counts = df_period["neighborhoods_analysis_boundaries"].value_counts(ascending=True)
fig_neigh = render_hbar(neigh_counts, f"New Businesses by Neighborhood ({period_label})")

# Optional tables
with st.expander("Show underlying tables"):
    st.write("Industry counts")
    st.dataframe(
        industry_counts.sort_values(ascending=False).rename_axis("Industry").reset_index(name="Count")
    )
    st.write("Neighborhood counts")
    st.dataframe(
        neigh_counts.sort_values(ascending=False).rename_axis("Neighborhood").reset_index(name="Count")
    )

# -------------------- PDF download --------------------
fig_industry_path = "industry_chart.png"
fig_neigh_path = "neighborhood_chart.png"
fig_industry.savefig(fig_industry_path, bbox_inches="tight")
fig_neigh.savefig(fig_neigh_path, bbox_inches="tight")

pdf_bytes = make_pdf(period_label, today_c, week_c, month_c, fig_industry_path, fig_neigh_path)
st.download_button(
    "Download PDF",
    data=pdf_bytes,
    file_name=f"SF_Business_Registrations_{period_choice.replace(' ', '_')}.pdf",
    mime="application/pdf",
)

st.caption("Data source: Registered Business Locations – San Francisco (g8m3-pdis) • Powered by Socrata SODA API")
