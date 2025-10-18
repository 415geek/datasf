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

# --- Builder credit with LinkedIn button (under title) ---
LINKEDIN_URL = "https://www.linkedin.com/in/lingyu-maxwell-lai"
st.markdown(
    f"""
<div style="display:flex;align-items:center;gap:10px;margin-top:-6px;margin-bottom:8px;">
  <div style="font-size:14px;color:#666;">
    Builded by <strong>Maxwell Lai</strong>
  </div>
  <a href="{LINKEDIN_URL}" target="_blank" title="LinkedIn: Maxwell Lai"
     style="display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;
            border-radius:4px;background:#0A66C2;">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg"
         alt="LinkedIn" width="12" height="12" style="filter: invert(1);" />
  </a>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("📊 Real-time data from DataSF")


# -------------------- Socrata config --------------------
SOC_DOMAIN = st.secrets.get("socrata", {}).get("domain", "data.sfgov.org")
DATASET_ID = st.secrets.get("socrata", {}).get("dataset_id", "g8m3-pdis")
APP_TOKEN = st.secrets.get("socrata", {}).get("app_token", None)
USERNAME = st.secrets.get("socrata", {}).get("username", None)
PASSWORD = st.secrets.get("socrata", {}).get("password", None)

BASE_URL = f"https://{SOC_DOMAIN}/resource/{DATASET_ID}.json"
HEADERS = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
AUTH = (USERNAME, PASSWORD) if USERNAME and PASSWORD else None


# -------------------- Helper utilities --------------------
def _dt_iso(d: date) -> str:
    return f"{d.isoformat()}T00:00:00.000"


def _with_app_token(params: dict) -> dict:
    if APP_TOKEN and "$$app_token" not in params:
        params = {**params, "$$app_token": APP_TOKEN}
    return params


def socrata_get(params: dict, timeout=45, max_retries=3):
    """GET wrapper with retries and token fallback."""
    global HEADERS
    params = _with_app_token(params)
    last_err = None
    used_token = "$$app_token" in params or ("X-App-Token" in HEADERS)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, headers=HEADERS, auth=AUTH, timeout=timeout)

            if resp.status_code == 429:
                wait = 2 ** attempt
                st.warning(f"⏳ Rate limited (HTTP 429). Retrying in {wait}s …")
                time.sleep(wait)
                continue

            if resp.status_code == 403 and "Invalid app_token" in (resp.text or "") and used_token:
                st.warning("⚠️ Invalid app_token. Retrying without token …")
                HEADERS = {k: v for k, v in HEADERS.items() if k.lower() != "x-app-token"}
                params.pop("$%24app_token", None)
                params.pop("$$app_token", None)
                used_token = False
                resp = requests.get(BASE_URL, params=params, headers=HEADERS, auth=AUTH, timeout=timeout)

            if not resp.ok:
                snippet = (resp.text or "")[:250].replace("\n", " ")
                st.error(f"🚫 HTTP {resp.status_code} · {snippet}")
                resp.raise_for_status()

            return resp
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 * attempt)
    raise last_err


BASIC_COLS = [
    "ttxid",
    "location_start_date",
    "naic_code_description",
    "neighborhoods_analysis_boundaries",
]


@st.cache_data(ttl=600, show_spinner=False)
def fetch_range(start_d: date, end_exclusive_d: date) -> pd.DataFrame:
    """Fetch rows safely, no $select to avoid column mismatch."""
    where = (
        f"location_start_date >= '{_dt_iso(start_d)}' AND "
        f"location_start_date < '{_dt_iso(end_exclusive_d)}'"
    )

    # Count first
    params_count = {"$select": "count(ttxid)", "$where": where}
    r = socrata_get(params_count, timeout=30)
    j = r.json()
    total = int(j[0].get("count_ttxid", 0)) if j else 0
    if total == 0:
        return pd.DataFrame(columns=BASIC_COLS + ["start_date"])

    all_rows = []
    limit = 1000
    for offset in range(0, total, limit):
        params = {
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

    for col in BASIC_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["naic_code_description"] = df["naic_code_description"].fillna("Unknown")
    df["neighborhoods_analysis_boundaries"] = df["neighborhoods_analysis_boundaries"].fillna(
        "Outside San Francisco"
    )

    df["start_date"] = pd.to_datetime(df.get("location_start_date"), errors="coerce").dt.date
    return df


def counts_block(df, today, sow, eow, som, eom):
    if df.empty:
        return 0, 0, 0
    return (
        int((df["start_date"] == today).sum()),
        int(((df["start_date"] >= sow) & (df["start_date"] < eow)).sum()),
        int(((df["start_date"] >= som) & (df["start_date"] < eom)).sum()),
    )


def render_hbar(counts: pd.Series, title: str):
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


def make_pdf(period_label, today_c, week_c, month_c, fig1_path, fig2_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "SF Business Registrations Report", ln=1, align="C")
    pdf.set_font("Helvetica", "", 12)

    period_label = period_label.replace("–", "-").replace("—", "-")

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


# -------------------- Neighborhood Emoji Map --------------------
NEIGHBORHOOD_EMOJI = {
    "Chinatown": "🐉",
    "Financial District/South Beach": "🏙️",
    "Mission": "🎨",
    "Sunset/Parkside": "🌅",
    "Richmond": "🌉",
    "North Beach": "🍝",
    "SoMa": "💼",
    "Downtown/Civic Center": "🏛️",
    "Outer Mission": "🏠",
    "Castro/Upper Market": "🏳️‍🌈",
    "Haight Ashbury": "🎸",
    "Marina": "⛵",
    "Presidio": "🌲",
    "Bayview Hunters Point": "⚙️",
    "Excelsior": "🛍️",
}


# -------------------- Date ranges --------------------
today = date.today()
start_of_week = today - timedelta(days=today.weekday())
start_of_next_week = start_of_week + timedelta(days=7)
start_of_month = today.replace(day=1)
start_of_next_month = (
    start_of_month.replace(month=start_of_month.month % 12 + 1, day=1)
    if start_of_month.month < 12
    else start_of_month.replace(year=start_of_month.year + 1, month=1, day=1)
)

earliest_needed = min(start_of_week, start_of_month)
latest_needed = max(start_of_next_week, start_of_next_month)

with st.spinner("Fetching live data from Socrata..."):
    superset_df = fetch_range(earliest_needed, latest_needed)

today_c, week_c, month_c = counts_block(
    superset_df, today, start_of_week, start_of_next_week, start_of_month, start_of_next_month
)

k1, k2, k3 = st.columns(3)
k1.metric("📅 New Businesses Today", today_c)
k2.metric("📈 New This Week", week_c)
k3.metric("🏢 New This Month", month_c)

st.markdown("### Explore by Period, Industry, and Neighborhood")
period_choice = st.radio("Time period", ["Today", "This Week", "This Month", "Custom"], index=0, horizontal=True)

# --- Safe custom date input handling ---
if period_choice == "Today":
    df_period = superset_df[superset_df["start_date"] == today]
    period_label = today.strftime("%b %d, %Y")
elif period_choice == "This Week":
    mask = (superset_df["start_date"] >= start_of_week) & (superset_df["start_date"] < start_of_next_week)
    df_period = superset_df[mask]
    period_label = f"{start_of_week.strftime('%b %d')} - {(start_of_next_week - timedelta(days=1)).strftime('%b %d, %Y')}"
elif period_choice == "This Month":
    mask = (superset_df["start_date"] >= start_of_month) & (superset_df["start_date"] < start_of_next_month)
    df_period = superset_df[mask]
    period_label = start_of_month.strftime("%B %Y")
else:
    default_start = today - timedelta(days=7)
    default_end = today
    date_range = st.date_input("Select date range", value=(default_start, default_end))
    if not (isinstance(date_range, tuple) and len(date_range) == 2):
        st.info("📆 Please select both a start and end date to display data.")
        st.stop()
    start_d, end_d = date_range
    end_exclusive = end_d + timedelta(days=1)
    with st.spinner("Fetching custom range..."):
        df_period = fetch_range(start_d, end_exclusive)
    period_label = f"{start_d.strftime('%b %d, %Y')} - {end_d.strftime('%b %d, %Y')}"

if df_period.empty:
    st.info("No business registrations found for the selected period.")
    st.stop()

# -------------------- Charts --------------------
st.markdown("#### 🏭 New Businesses by Industry")
industry_counts = df_period["naic_code_description"].value_counts(ascending=True)
fig_industry = render_hbar(industry_counts, f"New Businesses by Industry ({period_label})")

# Add emoji to neighborhood labels
df_period["neighborhoods_with_emoji"] = df_period["neighborhoods_analysis_boundaries"].apply(
    lambda n: f"{NEIGHBORHOOD_EMOJI.get(n, '📍')} {n}"
)

st.markdown("#### 🗺️ New Businesses by Neighborhood")
neigh_counts = df_period["neighborhoods_with_emoji"].value_counts(ascending=True)
fig_neigh = render_hbar(neigh_counts, f"New Businesses by Neighborhood ({period_label})")

# -------------------- View Details by Industry --------------------
st.markdown("### 🔍 View Details by Industry")
industry_options = list(industry_counts.index[::-1])
sel_industry = st.selectbox("Pick an industry to list all new registrations", options=industry_options, index=0)
detail_df = df_period[df_period["naic_code_description"] == sel_industry].copy()


def pick_col(df, candidates, new_name):
    for c in candidates:
        if c in df.columns:
            df[new_name] = df[c]
            return
    df[new_name] = np.nan


pick_col(detail_df, ["location_start_date"], "Start Date")
pick_col(detail_df, ["dba_name"], "DBA Name")
pick_col(detail_df, ["ownership_name", "owner_name"], "Owner/Legal Name")
pick_col(detail_df, ["certificate_number"], "Certificate #")
pick_col(detail_df, ["uniqueid", "ttxid"], "Record ID")
pick_col(detail_df, ["naics_code", "naic_code", "naics"], "NAICS Code")
pick_col(detail_df, ["naic_code_description", "naics_description"], "NAICS Description")
pick_col(detail_df, ["full_business_address", "street_address", "business_address"], "Address")
pick_col(detail_df, ["city"], "City")
pick_col(detail_df, ["state"], "State")
pick_col(detail_df, ["business_zip", "source_zipcode", "zip_code"], "ZIP")
pick_col(detail_df, ["neighborhoods_analysis_boundaries"], "Neighborhood")
pick_col(detail_df, ["business_corridor"], "Business Corridor")
pick_col(detail_df, ["business_location"], "Business Location (Geo)")

display_cols = [
    "Start Date",
    "DBA Name",
    "Owner/Legal Name",
    "Certificate #",
    "Record ID",
    "NAICS Code",
    "NAICS Description",
    "Address",
    "City",
    "State",
    "ZIP",
    "Neighborhood",
    "Business Corridor",
    "Business Location (Geo)",
]
display_cols = [c for c in display_cols if c in detail_df.columns]
detail_df = detail_df[display_cols]

st.write(f"**{sel_industry}** — {len(detail_df)} new registrations in the selected period")
st.dataframe(detail_df, use_container_width=True)

csv_bytes = detail_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Download CSV (Selected Industry Details)",
    data=csv_bytes,
    file_name=f"{sel_industry.replace(' ', '_')}_Details_{period_choice.replace(' ', '_')}.csv",
    mime="text/csv",
)

# -------------------- Optional tables --------------------
with st.expander("📋 Show underlying count tables"):
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
    "📄 Download PDF (Charts & KPIs)",
    data=pdf_bytes,
    file_name=f"SF_Business_Registrations_{period_choice.replace(' ', '_')}.pdf",
    mime="application/pdf",
)

st.caption("Data source: Registered Business Locations – San Francisco (g8m3-pdis) • Powered by Socrata SODA API")
