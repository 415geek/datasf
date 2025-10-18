import os
import io
from datetime import date, timedelta

import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF

# ---------- Page / App config ----------
st.set_page_config(page_title="SF Business Registrations Dashboard", layout="wide")
st.title("San Francisco Business Registrations Dashboard")
st.caption("Real-time insights from DataSF · Registered Business Locations (g8m3-pdis)")

# ---------- Secrets (Socrata) ----------
# Put these in .streamlit/secrets.toml or Streamlit Cloud Secrets
SOC_DOMAIN   = st.secrets.get("socrata", {}).get("domain", "data.sfgov.org")
DATASET_ID   = st.secrets.get("socrata", {}).get("dataset_id", "g8m3-pdis")
APP_TOKEN    = st.secrets.get("socrata", {}).get("app_token", None)
USERNAME     = st.secrets.get("socrata", {}).get("username", None)   # optional
PASSWORD     = st.secrets.get("socrata", {}).get("password", None)   # optional

BASE_URL = f"https://{SOC_DOMAIN}/resource/{DATASET_ID}.json"

HEADERS = {}
if APP_TOKEN:
    HEADERS["X-App-Token"] = APP_TOKEN

AUTH = None
if USERNAME and PASSWORD:
    AUTH = (USERNAME, PASSWORD)

# ---------- Helpers ----------
def _dt_iso(d: date) -> str:
    # Socrata understands ISO8601; append midnight for inclusive start and next day's midnight for exclusive end
    return f"{d.isoformat()}T00:00:00.000"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_range(start_d: date, end_exclusive_d: date) -> pd.DataFrame:
    """
    Fetch all rows with location_start_date in [start_d, end_exclusive_d)
    Handles paging (1000 row/chunk). Only pulls needed columns for speed.
    """
    where = (
        f"location_start_date >= '{_dt_iso(start_d)}' AND "
        f"location_start_date < '{_dt_iso(end_exclusive_d)}'"
    )
    select_cols = [
        "ttxid",
        "location_start_date",
        "naic_code_description",
        "neighborhoods_analysis_boundaries"
    ]
    # 1) Get count
    params_count = {
        "$select": "count(ttxid)",
        "$where": where
    }
    r = requests.get(BASE_URL, params=params_count, headers=HEADERS, auth=AUTH, timeout=30)
    r.raise_for_status()
    j = r.json()
    total = int(j[0].get("count_ttxid", 0)) if j else 0
    if total == 0:
        return pd.DataFrame(columns=select_cols + ["start_date"])

    # 2) Page through
    out = []
    limit = 1000
    for offset in range(0, total, limit):
        params = {
            "$select": ", ".join(select_cols),
            "$where": where,
            "$order": "location_start_date",
            "$limit": limit,
            "$offset": offset,
        }
        rr = requests.get(BASE_URL, params=params, headers=HEADERS, auth=AUTH, timeout=45)
        rr.raise_for_status()
        out.extend(rr.json())

    df = pd.DataFrame(out)
    if df.empty:
        return df

    # Normalize fields
    if "naic_code_description" not in df.columns:
        df["naic_code_description"] = np.nan
    if "neighborhoods_analysis_boundaries" not in df.columns:
        df["neighborhoods_analysis_boundaries"] = np.nan

    df["naic_code_description"] = df["naic_code_description"].fillna("Unknown")
    df["neighborhoods_analysis_boundaries"] = df["neighborhoods_analysis_boundaries"].fillna("Outside San Francisco")

    # Parse date -> date (not datetime)
    df["start_date"] = pd.to_datetime(df["location_start_date"], errors="coerce").dt.date
    return df

def counts_block(df: pd.DataFrame, today: date, sow: date, eow: date, som: date, eom: date):
    if df.empty:
        return 0, 0, 0
    today_c = int((df["start_date"] == today).sum())
    week_c  = int(((df["start_date"] >= sow) & (df["start_date"] < eow)).sum())
    month_c = int(((df["start_date"] >= som) & (df["start_date"] < eom)).sum())
    return today_c, week_c, month_c

def render_hbar(counts: pd.Series, title: str):
    fig = plt.figure(figsize=(8, max(4, 0.3 * len(counts))))
    ax = fig.add_subplot(1,1,1)
    y = np.arange(len(counts))
    ax.barh(y, counts.values)  # DO NOT set colors explicitly (per instruction)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of New Businesses")
    ax.set_title(title)
    st.pyplot(fig)
    return fig

def make_pdf(period_label: str, today_c: int, week_c: int, month_c: int, fig1_path: str, fig2_path: str) -> bytes:
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

# ---------- Date ranges (today / week / month) ----------
today = date.today()
start_of_today = today
start_of_tomorrow = today + timedelta(days=1)

# Week: Monday start
weekday = today.weekday()  # Monday=0
start_of_week = today - timedelta(days=weekday)
start_of_next_week = start_of_week + timedelta(days=7)

# Month
start_of_month = today.replace(day=1)
start_of_next_month = (start_of_month.replace(month=start_of_month.month % 12 + 1, day=1)
                       if start_of_month.month < 12
                       else start_of_month.replace(year=start_of_month.year + 1, month=1, day=1))

# Fetch once a superset covering week+month to populate metrics quickly
earliest = min(start_of_week, start_of_month)
latest   = max(start_of_next_week, start_of_next_month)

with st.spinner("Fetching live data from Socrata..."):
    superset = fetch_range(earliest, latest)

# ---------- KPI Metrics ----------
t_c, w_c, m_c = counts_block(superset, today, start_of_week, start_of_next_week, start_of_month, start_of_next_month)
c1, c2, c3 = st.columns(3)
c1.metric("New Businesses Today", t_c)
c2.metric("New This Week", w_c)
c3.metric("New This Month", m_c)

st.markdown("### Explore by Period, Industry, and Neighborhood")
period_choice = st.radio("Time period", ["Today", "This Week", "This Month", "Custom"], index=0, horizontal=True)

if period_choice == "Today":
    result_df = superset[superset["start_date"] == today]
    period_label = today.strftime("%b %d, %Y")
elif period_choice == "This Week":
    mask = (superset["start_date"] >= start_of_week) & (superset["start_date"] < start_of_next_week)
    result_df = superset[mask]
    period_label = f"{start_of_week.strftime('%b %d')} – { (start_of_next_week - timedelta(days=1)).strftime('%b %d, %Y') }"
elif period_choice == "This Month":
    mask = (superset["start_date"] >= start_of_month) & (superset["start_date"] < start_of_next_month)
    result_df = superset[mask]
    period_label = start_of_month.strftime("%B %Y")
else:
    default_start = today - timedelta(days=7)
    default_end   = today
    start_d, end_d = st.date_input("Select date range", value=(default_start, default_end))
    if isinstance(start_d, tuple):  # older Streamlit versions
        start_d, end_d = start_d
    end_exclusive = end_d + timedelta(days=1)
    with st.spinner("Fetching custom range..."):
        result_df = fetch_range(start_d, end_exclusive)
    period_label = f"{start_d.strftime('%b %d, %Y')} – {end_d.strftime('%b %d, %Y')}"

if result_df.empty:
    st.info("No business registrations found for the selected period.")
    st.stop()

# ---------- Industry & Neighborhood breakdown ----------
st.markdown("#### New Businesses by Industry")
industry_counts = result_df["naic_code_description"].value_counts(ascending=True)
fig_industry = render_hbar(industry_counts, f"New Businesses by Industry ({period_label})")

st.markdown("#### New Businesses by Neighborhood")
neigh_counts = result_df["neighborhoods_analysis_boundaries"].value_counts(ascending=True)
fig_neigh = render_hbar(neigh_counts, f"New Businesses by Neighborhood ({period_label})")

# ---------- Show tables (optional, expands on click) ----------
with st.expander("Show underlying tables"):
    st.write("Industry counts")
    st.dataframe(industry_counts.sort_values(ascending=False).rename_axis("Industry").reset_index(name="Count"))
    st.write("Neighborhood counts")
    st.dataframe(neigh_counts.sort_values(ascending=False).rename_axis("Neighborhood").reset_index(name="Count"))

# ---------- PDF download (charts + KPIs for the chosen period) ----------
# Save figs to temp PNGs
fig_industry_path = "industry_chart.png"
fig_neigh_path = "neighborhood_chart.png"
fig_industry.savefig(fig_industry_path, bbox_inches="tight")
fig_neigh.savefig(fig_neigh_path, bbox_inches="tight")

pdf_bytes = make_pdf(period_label, t_c, w_c, m_c, fig_industry_path, fig_neigh_path)
st.download_button("Download PDF", data=pdf_bytes, file_name=f"SF_Business_Registrations_{period_choice.replace(' ','_')}.pdf", mime="application/pdf")

st.caption("Data source: Registered Business Locations – San Francisco (g8m3-pdis) • Powered by Socrata SODA API")
