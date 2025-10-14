"""Streamlit Facility Dashboard (Provider Info)

Allows searching and viewing per-facility pages sourced from
`provider_info_combined.csv`. Maps processing months to actual quarters
based on 6-month delay pattern for staffing data.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


DATA_PATH = Path("provider_info_combined.csv")
MAPPING_PATH = Path("quarter_to_provider_mapping_examples.csv")

CANONICAL_COLUMNS = [
    "ccn",
    "provider_name",
    "state",
    "city",
    "county",
    "ownership_type",
    "avg_residents_per_day",
    "overall_rating",
    "reported_na_hrs_per_resident_per_day",
    "reported_lpn_hrs_per_resident_per_day",
    "reported_rn_hrs_per_resident_per_day",
    "reported_licensed_hrs_per_resident_per_day",
    "reported_total_nurse_hrs_per_resident_per_day",
    "weekend_total_nurse_hrs_per_resident_per_day",
    "weekend_rn_hrs_per_resident_per_day",
    "reported_pt_hrs_per_resident_per_day",
    "case_mix_na_hrs_per_resident_per_day",
    "case_mix_lpn_hrs_per_resident_per_day",
    "case_mix_rn_hrs_per_resident_per_day",
    "case_mix_total_nurse_hrs_per_resident_per_day",
    "case_mix_weekend_total_nurse_hrs_per_resident_per_day",
    "adjusted_na_hrs_per_resident_per_day",
    "adjusted_lpn_hrs_per_resident_per_day",
    "adjusted_rn_hrs_per_resident_per_day",
    "adjusted_total_nurse_hrs_per_resident_per_day",
    "adjusted_weekend_total_nurse_hrs_per_resident_per_day",
    "staffing_rating",
    "health_inspection_rating",
    "total_nursing_staff_turnover",
    "registered_nurse_turnover",
    "provider_changed_ownership_in_last_12_months",
    "processing_date",
]


def get_quarter_from_processing_month(proc_month: str) -> str | None:
    """
    Map processing month to actual quarter based on EXACT mappings from quarter_to_provider_mapping_examples.csv.
    The pattern is completely inconsistent - sometimes 3 months behind, sometimes 6, sometimes 9+.
    """
    if not proc_month or pd.isna(proc_month):
        return None
    
    try:
        year, month = map(int, proc_month.split('-'))
        
        # Create exact lookup table based on the mapping examples file
        # Format: (year, month) -> quarter
        # For conflicts, prioritize the most recent quarter
        exact_mappings = {
            # 2017Q4
            (2018, 4): "Q4 2017", (2018, 5): "Q4 2017", (2018, 6): "Q4 2017",
            # 2018Q1  
            (2018, 7): "Q1 2018", (2018, 8): "Q1 2018", (2018, 9): "Q1 2018",
            # 2018Q2
            (2018, 10): "Q2 2018", (2018, 11): "Q2 2018", (2018, 12): "Q2 2018",
            # 2018Q3
            (2019, 1): "Q3 2018", (2019, 2): "Q3 2018", (2019, 3): "Q3 2018",
            # 2018Q4
            (2019, 4): "Q4 2018", (2019, 5): "Q4 2018", (2019, 6): "Q4 2018",
            # 2019Q1
            (2019, 7): "Q1 2019", (2019, 8): "Q1 2019", (2019, 9): "Q1 2019",
            # 2019Q2
            (2019, 10): "Q2 2019", (2019, 11): "Q2 2019",
            # 2019Q3
            (2020, 1): "Q3 2019", (2020, 2): "Q3 2019", (2020, 3): "Q3 2019",
            # 2019Q4 (spans 6 months!)
            (2020, 4): "Q4 2019", (2020, 5): "Q4 2019", (2020, 6): "Q4 2019", 
            (2020, 7): "Q4 2019", (2020, 8): "Q4 2019", (2020, 9): "Q4 2019",
            # 2020Q2 (only 2 months!)
            (2020, 10): "Q2 2020", (2020, 11): "Q2 2020",
            # 2020Q3
            (2021, 1): "Q3 2020", (2021, 2): "Q3 2020", (2021, 3): "Q3 2020",
            # 2020Q4
            (2021, 4): "Q4 2020", (2021, 5): "Q4 2020", (2021, 6): "Q4 2020",
            # 2021Q1
            (2021, 7): "Q1 2021", (2021, 8): "Q1 2021", (2021, 9): "Q1 2021",
            # 2021Q2
            (2021, 10): "Q2 2021", (2021, 11): "Q2 2021",
            # 2021Q3
            (2022, 1): "Q3 2021", (2022, 2): "Q3 2021", (2022, 3): "Q3 2021",
            # 2021Q4
            (2022, 4): "Q4 2021", (2022, 5): "Q4 2021", (2022, 6): "Q4 2021",
            # 2022Q1
            (2022, 7): "Q1 2022", (2022, 8): "Q1 2022", (2022, 9): "Q1 2022",
            # 2022Q2 - FIXED: should map to 2022Q2, not Q1 2025
            (2022, 10): "Q2 2022", (2022, 11): "Q2 2022",
            # 2022Q3
            (2023, 1): "Q3 2022", (2023, 2): "Q3 2022", (2023, 3): "Q3 2022",
            # 2022Q4
            (2023, 4): "Q4 2022", (2023, 5): "Q4 2022", (2023, 6): "Q4 2022",
            # 2023Q1
            (2023, 7): "Q1 2023", (2023, 8): "Q1 2023", (2023, 9): "Q1 2023",
            # 2023Q2 (spans 9 months!)
            (2023, 10): "Q2 2023", (2023, 11): "Q2 2023",
            (2024, 1): "Q2 2023", (2024, 2): "Q2 2023", (2024, 3): "Q2 2023",
            (2024, 4): "Q2 2023", (2024, 5): "Q2 2023", (2024, 6): "Q2 2023",
            # 2023Q3 (spans 6 months! - CONFLICT with Q2 2023, prioritize Q3)
            (2024, 1): "Q3 2023", (2024, 2): "Q3 2023", (2024, 3): "Q3 2023",
            (2024, 4): "Q3 2023", (2024, 5): "Q3 2023", (2024, 6): "Q3 2023",
            # 2024Q1
            (2024, 7): "Q1 2024", (2024, 8): "Q1 2024", (2024, 9): "Q1 2024",
            # 2024Q2
            (2024, 10): "Q2 2024", (2024, 11): "Q2 2024",
            # 2024Q3 (only 2 months!)
            (2025, 2): "Q3 2024", (2025, 3): "Q3 2024",
            # 2024Q4
            (2025, 4): "Q4 2024", (2025, 5): "Q4 2024", (2025, 6): "Q4 2024",
            # 2025Q1 - CORRECTED: 2025-07 maps to 2025Q1, not 2024Q3
            (2025, 7): "Q1 2025",
        }
        
        return exact_mappings.get((year, month))
        
    except (ValueError, IndexError):
        return None


def get_processing_months_for_quarter(quarter: str) -> list[str]:
    """
    Get the processing months that contain data for a given quarter.
    """
    if not quarter or len(quarter) < 6:
        return []
    
    try:
        # Parse "2018Q1" format
        if len(quarter) == 6 and quarter[4] == 'Q':
            year = int(quarter[:4])  # 2018
            q_num = int(quarter[5])  # 1
        else:
            parts = quarter.split()
            if len(parts) != 2:
                return []
            
            q_part = parts[0]  # "Q1"
            year = int(parts[1])  # 2018
            q_num = int(q_part[1])  # 1
        
        if q_num == 1:
            return [f"{year}-04", f"{year}-05", f"{year}-06"]
        elif q_num == 2:
            return [f"{year}-07", f"{year}-08", f"{year}-09"]
        elif q_num == 3:
            return [f"{year}-10", f"{year}-11", f"{year}-12"]
        elif q_num == 4:
            return [f"{year+1}-01", f"{year+1}-02", f"{year+1}-03"]
        else:
            return []
    except (ValueError, IndexError):
        return []


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Missing dataset: {DATA_PATH}")
        st.stop()
    if not MAPPING_PATH.exists():
        st.error(f"Missing mapping file: {MAPPING_PATH}")
        st.stop()
    
    # Load the main data
    df = pd.read_csv(DATA_PATH, dtype={"ccn": "string"}, low_memory=False)
    
    # Load the mapping file to get correct quarters
    mapping_df = pd.read_csv(MAPPING_PATH)
    
    # Create a lookup table from processing_date to quarter
    mapping_df['processing_date'] = pd.to_datetime(mapping_df['processing_date'])
    # Create a mapping that uses the exact mappings from the mapping file
    # The mapping file shows the correct quarter for each processing_date + HPRD combination
    date_to_quarter = {}
    for _, row in mapping_df.iterrows():
        date = row['processing_date']
        quarter = row['quarter']
        hprd = row['lite_total_nurse_hprd']
        
        # Create a key that combines date and HPRD value to handle conflicts
        key = (date, round(hprd, 3))  # Round to 3 decimal places to handle floating point precision
        
        # Handle conflicts by prioritizing Q3 2023 over Q2 2023 for 2024-01 through 2024-06
        if key in date_to_quarter:
            existing_quarter = date_to_quarter[key]
            if existing_quarter == '2023Q2' and quarter == '2023Q3':
                date_to_quarter[key] = quarter  # Prioritize Q3 2023
            elif existing_quarter == '2023Q3' and quarter == '2023Q2':
                pass  # Keep Q3 2023
            else:
                date_to_quarter[key] = quarter  # Use the new quarter for other conflicts
        else:
            date_to_quarter[key] = quarter
    
    # Ensure presence of columns; if missing, create empty
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    
    # Parse dates and add quarter mapping
    if "processing_date" in df.columns:
        df["processing_date"] = pd.to_datetime(df["processing_date"], errors="coerce")
        # Precompute display month label (YYYY-MM) for selection
        df["proc_month"] = df["processing_date"].dt.to_period("M").astype(str)
        # Add quarter mapping using the mapping file
        # Create a function to map based on processing_date and HPRD values
        def get_quarter(row):
            date = row['processing_date']
            
            # First, try to find any mapping for this date and prioritize Q3 2023 over Q2 2023
            quarters_for_date = []
            for (mapping_date, mapping_hprd), quarter in date_to_quarter.items():
                if mapping_date == date:
                    quarters_for_date.append(quarter)
            
            if quarters_for_date:
                # If we have both Q2 2023 and Q3 2023, prioritize Q3 2023
                if '2023Q3' in quarters_for_date and '2023Q2' in quarters_for_date:
                    return '2023Q3'
                # Otherwise, return the first (or most common) quarter
                return quarters_for_date[0]
            
            return None
        
        df["quarter"] = df.apply(get_quarter, axis=1)
        # Filter to only include quarters from Q4 2017 onwards
        df = df[df["quarter"] >= "2017Q4"]
    
    # Clean text
    for col in ["provider_name", "state", "city", "county", "ownership_type"]:
        df[col] = df[col].astype("string").str.strip()
    
    # Cast ratings to integers for display/metrics
    df["overall_rating"] = pd.to_numeric(df.get("overall_rating"), errors="coerce").astype("Int64")
    
    # Cast staffing/case-mix columns to numeric for charts
    numeric_cols = [
        "avg_residents_per_day",
        "reported_na_hrs_per_resident_per_day",
        "reported_lpn_hrs_per_resident_per_day",
        "reported_rn_hrs_per_resident_per_day",
        "reported_licensed_hrs_per_resident_per_day",
        "reported_total_nurse_hrs_per_resident_per_day",
        "reported_pt_hrs_per_resident_per_day",
        "case_mix_na_hrs_per_resident_per_day",
        "case_mix_lpn_hrs_per_resident_per_day",
        "case_mix_rn_hrs_per_resident_per_day",
        "case_mix_total_nurse_hrs_per_resident_per_day",
        "adjusted_na_hrs_per_resident_per_day",
        "adjusted_lpn_hrs_per_resident_per_day",
        "adjusted_rn_hrs_per_resident_per_day",
        "adjusted_total_nurse_hrs_per_resident_per_day",
        "staffing_rating",
        "health_inspection_rating",
    ]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce")
    return df


def facility_search(df: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    # Build latest snapshot per CCN for search
    latest = (
        df.sort_values("processing_date")
        .groupby("ccn", as_index=False)
        .tail(1)
    )
    latest = latest.sort_values(["state", "city", "provider_name"])  # UX-friendly ordering

    st.subheader("Find a facility")
    col1, col2 = st.columns([1, 2])
    with col1:
        states = ["All"] + sorted(latest["state"].dropna().astype(str).unique().tolist())
        sel_state = st.selectbox("State", options=states)
    if sel_state != "All":
        latest = latest[latest["state"] == sel_state]

    with col2:
        options = latest[["ccn", "provider_name", "city", "state"]].copy()
        options["label"] = options.apply(
            lambda r: f"{str(r['ccn'])} — {str(r['provider_name'])} ({str(r['city'])}, {str(r['state'])})",
            axis=1,
        )
        labels = options["label"].tolist()
        label_to_ccn = dict(zip(labels, options["ccn"].astype("string")))
        sel_label = st.selectbox("Facility (by CCN)", options=labels if labels else [""], index=0 if labels else None)
        selected_ccn = label_to_ccn.get(sel_label)

    # Homepage should only show search (no data table)
    return selected_ccn, latest


def format_metric_table_value(value: float | int | str | None) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        f = float(value)
        return f"{f:.3f}"
    except Exception:
        return str(value)


def build_quarter_comparison_row(row: pd.Series) -> pd.DataFrame:
    reported = {
        "NA": row.get("reported_na_hrs_per_resident_per_day"),
        "LPN": row.get("reported_lpn_hrs_per_resident_per_day"),
        "RN": row.get("reported_rn_hrs_per_resident_per_day"),
        "Licensed": row.get("reported_licensed_hrs_per_resident_per_day"),
        "Total": row.get("reported_total_nurse_hrs_per_resident_per_day"),
        "PT": row.get("reported_pt_hrs_per_resident_per_day"),
    }

    case_mix = {
        "NA": row.get("case_mix_na_hrs_per_resident_per_day"),
        "LPN": row.get("case_mix_lpn_hrs_per_resident_per_day"),
        "RN": row.get("case_mix_rn_hrs_per_resident_per_day"),
        "Licensed": None if pd.isna(row.get("case_mix_rn_hrs_per_resident_per_day")) and pd.isna(row.get("case_mix_lpn_hrs_per_resident_per_day")) else (
            (row.get("case_mix_rn_hrs_per_resident_per_day") or 0)
            + (row.get("case_mix_lpn_hrs_per_resident_per_day") or 0)
        ),
        "Total": row.get("case_mix_total_nurse_hrs_per_resident_per_day"),
        "PT": None,  # no expected/PT analogue typically
    }

    records: list[dict] = []
    for metric in ["NA", "LPN", "RN", "Licensed", "Total", "PT"]:
        r = reported.get(metric)
        c = case_mix.get(metric)
        delta = None
        pct = None
        if r is not None and not pd.isna(r) and c is not None and not pd.isna(c):
            try:
                delta = float(r) - float(c)
                pct = (float(r) / float(c) - 1.0) if float(c) != 0 else None
            except Exception:
                delta = None
                pct = None
        records.append(
            {
                "metric": metric,
                "reported": r,
                "case_mix_expected": c,
                "delta": delta,
                "delta_pct": pct,
            }
        )
    out = pd.DataFrame(records)
    for col in ["reported", "case_mix_expected", "delta"]:
        out[col] = out[col].apply(format_metric_table_value)
    out["delta_pct"] = out["delta_pct"].apply(
        lambda v: "" if v is None or pd.isna(v) else f"{v*100:.1f}%"
    )
    return out


def show_facility_page(df: pd.DataFrame, selected_ccn: str) -> None:
    fac = df[df["ccn"].astype("string") == selected_ccn].copy()
    if fac.empty:
        st.warning("No data for selected CCN.")
        return
    fac = fac.sort_values("processing_date")

    left, right = st.columns([1, 0.001])
    with left:
        st.subheader(f"{fac['provider_name'].iloc[-1]} ({selected_ccn})")
        st.caption(
            f"{fac['city'].iloc[-1]}, {fac['state'].iloc[-1]} • {fac['ownership_type'].iloc[-1] or ''}"
        )

        # Latest snapshot (single horizontal row) - REMOVED per user request

        # Select a specific quarter for staffing data
        quarters = sorted(fac["quarter"].dropna().unique().tolist(), key=lambda x: (int(x[:4]), int(x[5])))
        sel_quarter = st.selectbox("Quarter (staffing data)", options=quarters, index=(len(quarters) - 1) if quarters else 0)

        # Get processing months for the selected quarter
        proc_months = get_processing_months_for_quarter(sel_quarter)
        row_q = fac[fac["proc_month"].isin(proc_months)].tail(1)  # Get latest record for this quarter
        if not row_q.empty:
            # Reported vs Case-Mix chart (PBJ style) for Total, RN, CNA
            r = row_q.iloc[0]
            facility_name = r.get("provider_name", "Unknown Facility")
            categories = ["Total", "RN", "CNA"]
            reported_values: list[float] = []
            case_mix_values: list[float] = []
            available_categories: list[str] = []

            for cat in categories:
                if cat == "Total":
                    rep_val = r.get("reported_total_nurse_hrs_per_resident_per_day")
                    cm_val = r.get("case_mix_total_nurse_hrs_per_resident_per_day")
                elif cat == "RN":
                    rep_val = r.get("reported_rn_hrs_per_resident_per_day")
                    cm_val = r.get("case_mix_rn_hrs_per_resident_per_day")
                else:  # CNA
                    rep_val = r.get("reported_na_hrs_per_resident_per_day")
                    cm_val = r.get("case_mix_na_hrs_per_resident_per_day")

                has_rep = pd.notna(rep_val) and float(rep_val) > 0
                has_cm = pd.notna(cm_val) and float(cm_val) > 0
                if has_rep or has_cm:
                    available_categories.append(cat)
                    reported_values.append(float(rep_val) if has_rep else 0.0)
                    case_mix_values.append(float(cm_val) if has_cm else 0.0)

            if any(v > 0 for v in case_mix_values):
                facility_name = str(fac["provider_name"].iloc[-1])
                fig_pbj = go.Figure()

                # Reported (Blue)
                fig_pbj.add_trace(
                    go.Bar(
                        x=available_categories,
                        y=reported_values,
                        name="Reported",
                        marker_color="blue",
                        hovertemplate="<b>%{x}</b><br>Reported: %{y:.2f} HPRD<extra></extra>",
                    )
                )
                # Case-Mix (Red)
                fig_pbj.add_trace(
                    go.Bar(
                        x=available_categories,
                        y=case_mix_values,
                        name="Case-Mix",
                        marker_color="red",
                        hovertemplate="<b>%{x}</b><br>Case-Mix: %{y:.2f} HPRD<extra></extra>",
                    )
                )

                fig_pbj.update_layout(
                    title=dict(
                        text=(
                            f"<span style='color: blue;'>Reported</span> vs. "
                            f"<span style='color: red;'>Case-Mix (Expected)</span><br>"
                            f"<span style='font-weight: normal;'>{facility_name}, {sel_quarter}</span>"
                        ),
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16),
                    ),
                    xaxis_title="",
                    yaxis_title="Hours per Resident Day",
                    height=280,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=50, r=50, t=100, b=50),
                    barmode="group",
                    bargap=0.1,
                    bargroupgap=0.05,
                    template="plotly_white",
                )

                # Delta annotations above bars
                for i, cat in enumerate(available_categories):
                    rep = reported_values[i]
                    cm = case_mix_values[i]
                    if cm > 0:
                        delta = rep - cm
                        pct = (rep / cm - 1.0) * 100.0
                        ymax = max(rep, cm)
                        gap_color = (
                            "rgba(33, 150, 243, 0.9)" if delta >= 0 else "rgba(244, 67, 54, 0.9)"
                        )
                        fig_pbj.add_annotation(
                            text=f"Δ: {delta:.2f} HPRD ({pct:.1f}%)",
                            x=cat,
                            y=ymax + 0.4,
                            showarrow=False,
                            font=dict(size=8, color="white"),
                            align="center",
                            bgcolor=gap_color,
                            bordercolor="rgba(0,0,0,0.1)",
                            borderwidth=1,
                        )

                st.plotly_chart(fig_pbj, use_container_width=True)

            # Methodology text mirroring PBJ verbiage (condensed)
            st.markdown(
                "_Case‑Mix (Expected) reflects resident acuity. Reported totals are compared against expected totals derived from CM/Expected hours per resident per day. Positive deltas indicate reported exceeds expected; negatives indicate under‑reported relative to expected._"
            )

            # Ratings methodology with footnotes
            with st.expander("Ratings Methodology & Footnotes", expanded=False):
                st.markdown("""
                **Star Ratings Methodology:**
                
                CMS assigns each nursing home a rating from 1 to 5 stars based on three domains:
                - **Overall Rating**: Combined score from all three domains
                - **Staffing Rating**: Based on reported staffing levels
                - **Health Inspection Rating**: Based on health inspection results
                
                **Footnote Codes:**
                - **1**: Newly certified nursing home with less than 12-15 months of data available
                - **2**: Not enough data available to calculate a star rating
                - **6**: Facility did not submit staffing data or data didn't meet criteria
                - **9**: Number of residents too small to report
                - **10**: Data missing or not submitted
                - **12**: No staffing data submitted, high days without RN, or unverified data
                - **13**: Results based on shorter time period than required
                - **14**: Not required to submit SNF Quality Reporting Program data
                - **18**: Not rated due to serious quality issues (Special Focus Facility)
                - **19**: Individual quarter scores not reported
                """)



        # Total Staffing Chart: Reported vs Case-Mix vs Adjusted
        total_cols = [
            "quarter",
            "processing_date",
                "reported_total_nurse_hrs_per_resident_per_day",
                "case_mix_total_nurse_hrs_per_resident_per_day",
                "adjusted_total_nurse_hrs_per_resident_per_day",
        ]
        total_df = fac[total_cols].dropna(subset=["quarter"]).copy()
        # Group by quarter and take the latest processing date per quarter
        total_df = total_df.sort_values("processing_date").groupby("quarter").last().reset_index()
        # Add file information for tooltips
        total_df["file_info"] = total_df["processing_date"].dt.strftime("%Y-%m")
        # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
        total_df["quarter_label"] = total_df["quarter"].apply(lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) else "")
        
        # Create the chart
        fig_total = go.Figure()
        
        # Add Reported Total line
        fig_total.add_trace(go.Scatter(
            x=total_df["quarter_label"],
            y=total_df["reported_total_nurse_hrs_per_resident_per_day"],
            mode='lines+markers',
            name='Reported Total',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            customdata=total_df["file_info"],
            hovertemplate="<b>Reported Total</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
        ))
        
        # Add Case-Mix Total line
        fig_total.add_trace(go.Scatter(
            x=total_df["quarter_label"],
            y=total_df["case_mix_total_nurse_hrs_per_resident_per_day"],
            mode='lines+markers',
            name='Case-Mix Total',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            customdata=total_df["file_info"],
            hovertemplate="<b>Case-Mix Total</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
        ))
        
        # Add Adjusted Total line (only for quarters that have it)
        adjusted_total_data = total_df.dropna(subset=["adjusted_total_nurse_hrs_per_resident_per_day"])
        if not adjusted_total_data.empty:
            fig_total.add_trace(go.Scatter(
                x=adjusted_total_data["quarter_label"],
                y=adjusted_total_data["adjusted_total_nurse_hrs_per_resident_per_day"],
                mode='lines+markers',
                name='Adjusted Total',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=6),
                customdata=adjusted_total_data["file_info"],
                hovertemplate="<b>Adjusted Total</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
            ))
        
        fig_total.update_layout(
            title=dict(
                text="Total Staffing: Reported vs Case-Mix vs Adjusted (by quarter)",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Quarter",
            yaxis_title="Hours per Resident Day",
            xaxis=dict(
                tickangle=45,
                nticks=len(total_df['quarter_label'].unique())
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        st.plotly_chart(fig_total, use_container_width=True)


        # RN Staffing Chart: Reported vs Case-Mix vs Adjusted
        rn_cols = [
            "quarter",
            "processing_date",
            "reported_rn_hrs_per_resident_per_day",
            "case_mix_rn_hrs_per_resident_per_day",
            "adjusted_rn_hrs_per_resident_per_day",
        ]
        rn_df = fac[rn_cols].dropna(subset=["quarter"]).copy()
        # Group by quarter and take the latest processing date per quarter
        rn_df = rn_df.sort_values("processing_date").groupby("quarter").last().reset_index()
        # Add file information for tooltips
        rn_df["file_info"] = rn_df["processing_date"].dt.strftime("%Y-%m")
        # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
        rn_df["quarter_label"] = rn_df["quarter"].apply(lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) else "")
        
        # Create the chart
        fig_rn = go.Figure()
        
        # Add Reported RN line
        fig_rn.add_trace(go.Scatter(
            x=rn_df["quarter_label"],
            y=rn_df["reported_rn_hrs_per_resident_per_day"],
            mode='lines+markers',
            name='Reported RN',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            customdata=rn_df["file_info"],
            hovertemplate="<b>Reported RN</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
        ))
        
        # Add Case-Mix RN line
        fig_rn.add_trace(go.Scatter(
            x=rn_df["quarter_label"],
            y=rn_df["case_mix_rn_hrs_per_resident_per_day"],
            mode='lines+markers',
            name='Case-Mix RN',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            customdata=rn_df["file_info"],
            hovertemplate="<b>Case-Mix RN</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
        ))
        
        # Add Adjusted RN line (only for quarters that have it)
        adjusted_rn_data = rn_df.dropna(subset=["adjusted_rn_hrs_per_resident_per_day"])
        if not adjusted_rn_data.empty:
            fig_rn.add_trace(go.Scatter(
                x=adjusted_rn_data["quarter_label"],
                y=adjusted_rn_data["adjusted_rn_hrs_per_resident_per_day"],
                mode='lines+markers',
                name='Adjusted RN',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=6),
                customdata=adjusted_rn_data["file_info"],
                hovertemplate="<b>Adjusted RN</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
            ))
        
        fig_rn.update_layout(
            title=dict(
                text="RN Staffing: Reported vs Case-Mix vs Adjusted (by quarter)",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Quarter",
            yaxis_title="Hours per Resident Day",
            xaxis=dict(
                tickangle=45,
                nticks=len(rn_df['quarter_label'].unique())
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        st.plotly_chart(fig_rn, use_container_width=True)

        # Nurse Aide Staffing Chart: Reported vs Case-Mix vs Adjusted
        cna_cols = [
            "quarter",
            "processing_date",
            "reported_na_hrs_per_resident_per_day",
            "case_mix_na_hrs_per_resident_per_day",
            "adjusted_na_hrs_per_resident_per_day",
        ]
        cna_df = fac[cna_cols].dropna(subset=["quarter"]).copy()
        # Group by quarter and take the latest processing date per quarter
        cna_df = cna_df.sort_values("processing_date").groupby("quarter").last().reset_index()
        # Add file information for tooltips
        cna_df["file_info"] = cna_df["processing_date"].dt.strftime("%Y-%m")
        # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
        cna_df["quarter_label"] = cna_df["quarter"].apply(lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) else "")
        
        # Create the chart
        fig_cna = go.Figure()
        
        # Add Reported Nurse Aide line
        fig_cna.add_trace(go.Scatter(
            x=cna_df["quarter_label"],
            y=cna_df["reported_na_hrs_per_resident_per_day"],
            mode='lines+markers',
            name='Reported Nurse Aide',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            customdata=cna_df["file_info"],
            hovertemplate="<b>Reported Nurse Aide</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
        ))
        
        # Add Case-Mix Nurse Aide line
        fig_cna.add_trace(go.Scatter(
            x=cna_df["quarter_label"],
            y=cna_df["case_mix_na_hrs_per_resident_per_day"],
            mode='lines+markers',
            name='Case-Mix Nurse Aide',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            customdata=cna_df["file_info"],
            hovertemplate="<b>Case-Mix Nurse Aide</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
        ))
        
        # Add Adjusted Nurse Aide line (only for quarters that have it)
        adjusted_cna_data = cna_df.dropna(subset=["adjusted_na_hrs_per_resident_per_day"])
        if not adjusted_cna_data.empty:
            fig_cna.add_trace(go.Scatter(
                x=adjusted_cna_data["quarter_label"],
                y=adjusted_cna_data["adjusted_na_hrs_per_resident_per_day"],
                mode='lines+markers',
                name='Adjusted Nurse Aide',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=6),
                customdata=adjusted_cna_data["file_info"],
                hovertemplate="<b>Adjusted Nurse Aide</b><br>Quarter: %{x}<br>HPRD: %{y:.3f}<br>File: %{customdata}<extra></extra>"
            ))
        
        fig_cna.update_layout(
            title=dict(
                text="Nurse Aide Staffing: Reported vs Case-Mix vs Adjusted (by quarter)",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Quarter",
            yaxis_title="Hours per Resident Day",
            xaxis=dict(
                tickangle=45,
                nticks=len(cna_df['quarter_label'].unique())
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        st.plotly_chart(fig_cna, use_container_width=True)


        # MDS Census by quarter
        census_df = fac[[
            "quarter",
            "processing_date",
            "avg_residents_per_day",
        ]].dropna(subset=["quarter"]).copy()
        # Group by quarter and take the latest processing date per quarter
        census_df = census_df.sort_values("processing_date").groupby("quarter").last().reset_index()
        if not census_df.empty:
            # Add file information for tooltips
            census_df["file_info"] = census_df["processing_date"].dt.strftime("%Y-%m")
            # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
            census_df["quarter_label"] = census_df["quarter"].apply(lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) else "")
            fig_census = px.line(
                census_df,
                x="quarter_label",
                y="avg_residents_per_day",
                markers=True,
                title="MDS Census by Quarter",
                custom_data=["file_info"],
            )
            # Update tooltips to include file information
            fig_census.update_traces(
                hovertemplate="<b>Census</b><br>Quarter: %{x}<br>Residents: %{y:.1f}<br>File: %{customdata[0]}<extra></extra>"
            )
            fig_census.update_layout(
                title_x=0.5,
                yaxis_title="Average Residents per Day",
                xaxis=dict(
                    tickangle=45,
                    nticks=len(census_df['quarter_label'].unique())  # Show all quarters
                )
            )
            st.plotly_chart(fig_census, use_container_width=True)

        # Delta over time: Reported Total - Case-Mix Total (by quarter)
        delta_df = fac[[
            "quarter",
            "processing_date",
            "reported_total_nurse_hrs_per_resident_per_day",
            "case_mix_total_nurse_hrs_per_resident_per_day",
        ]].dropna(subset=["quarter"])
        if not delta_df.empty:
            delta_df = delta_df.copy()
            # Group by quarter and take the latest processing date per quarter
            delta_df = delta_df.sort_values("processing_date").groupby("quarter").last().reset_index()
            # Add file information for tooltips
            delta_df["file_info"] = delta_df["processing_date"].dt.strftime("%Y-%m")
            # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
            delta_df["quarter_label"] = delta_df["quarter"].apply(lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) else "")
            delta_df["delta"] = (
                delta_df["reported_total_nurse_hrs_per_resident_per_day"]
                - delta_df["case_mix_total_nurse_hrs_per_resident_per_day"]
            )
            # Sort by quarter chronologically
            delta_df = delta_df.sort_values("quarter", key=lambda x: x.str[:4].astype(int) * 10 + x.str[5].astype(int))
            fig_delta = px.line(
                delta_df,
                x="quarter_label",
                y="delta",
                markers=True,
                title="Delta (Reported Total − Case‑Mix Total) over time (by quarter)",
                custom_data=["file_info"],
            )
            # Update tooltips to include file information
            fig_delta.update_traces(
                hovertemplate="<b>Delta</b><br>Quarter: %{x}<br>Delta: %{y:.3f}<br>File: %{customdata[0]}<extra></extra>"
            )
            fig_delta.update_layout(
                title_x=0.5, 
                yaxis_title="Hours per resident per day",
                xaxis=dict(
                    tickangle=45,
                    nticks=len(delta_df['quarter_label'].unique())
                )
            )
            # Emphasize zero baseline
            try:
                fig_delta.add_hline(y=0, line_width=3, line_color="#2c2c2c", layer="above")
            except Exception:
                pass
            st.plotly_chart(fig_delta, use_container_width=True)

            # Percentage delta over time
            pct_df = delta_df.copy()
            pct_df = pct_df[pct_df["case_mix_total_nurse_hrs_per_resident_per_day"] != 0]
            if not pct_df.empty:
                pct_df["pct_delta"] = (
                    pct_df["reported_total_nurse_hrs_per_resident_per_day"]
                    / pct_df["case_mix_total_nurse_hrs_per_resident_per_day"]
                    - 1.0
                ) * 100.0
                # Sort by quarter chronologically
                pct_df = pct_df.sort_values("quarter", key=lambda x: x.str[:4].astype(int) * 10 + x.str[5].astype(int))
                fig_pct = px.line(
                    pct_df,
                    x="quarter_label",
                    y="pct_delta",
                    markers=True,
                    title="Percent Delta (Reported Total vs Case‑Mix Total) over time (by quarter)",
                    custom_data=["file_info"],
                )
                # Update tooltips to include file information
                fig_pct.update_traces(
                    hovertemplate="<b>Percent Delta</b><br>Quarter: %{x}<br>Percent: %{y:.1f}%<br>File: %{customdata[0]}<extra></extra>"
                )
                fig_pct.update_layout(
                    title_x=0.5, 
                    yaxis_title="Percent",
                    xaxis=dict(
                        tickangle=45,
                        nticks=len(pct_df['quarter_label'].unique())
                    )
                )
                # Emphasize zero baseline
                try:
                    fig_pct.add_hline(y=0, line_width=3, line_color="#2c2c2c", layer="above")
                except Exception:
                    pass
                st.plotly_chart(fig_pct, use_container_width=True)

        # Adjusted staffing (Case-Mix / Expected) by quarter
        adjusted_cols = [
            "quarter",
            "processing_date",
            "adjusted_na_hrs_per_resident_per_day",
            "adjusted_lpn_hrs_per_resident_per_day",
            "adjusted_rn_hrs_per_resident_per_day",
            "adjusted_total_nurse_hrs_per_resident_per_day",
        ]
        adj_df = fac[adjusted_cols].dropna(subset=["quarter"]).copy()
        # Group by quarter and take the latest processing date per quarter
        adj_df = adj_df.sort_values("processing_date").groupby("quarter").last().reset_index()
        # Add file information for tooltips
        adj_df["file_info"] = adj_df["processing_date"].dt.strftime("%Y-%m")
        adj_long = adj_df.melt(
            id_vars=["quarter", "file_info"],
            value_vars=adjusted_cols[2:],  # Exclude quarter and processing_date columns
            var_name="metric",
            value_name="value",
        )
        adj_long["metric"] = adj_long["metric"].map(
            {
                "adjusted_na_hrs_per_resident_per_day": "Adjusted NA",
                "adjusted_lpn_hrs_per_resident_per_day": "Adjusted LPN",
                "adjusted_rn_hrs_per_resident_per_day": "Adjusted RN",
                "adjusted_total_nurse_hrs_per_resident_per_day": "Adjusted Total",
            }
        )
        fig_adj = px.line(
            adj_long,
            x="quarter",
            y="value",
            color="metric",
            markers=True,
            title="Adjusted staffing over time (by quarter)",
            custom_data=["file_info"],
        )
        # Update tooltips to include file information
        fig_adj.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Quarter: %{x}<br>Value: %{y:.3f}<br>File: %{customdata[0]}<extra></extra>"
        )
        fig_adj.update_layout(
            title_x=0.5,
            xaxis=dict(
                tickangle=45,
                nticks=len(adj_long['quarter'].unique())  # Show all quarters
            )
        )
        st.plotly_chart(fig_adj, use_container_width=True)

        # Ratings over time (Overall, Staffing, Health Inspection) by quarter
        ratings_df = fac[[
            "quarter",
            "processing_date",
            "overall_rating",
            "staffing_rating",
            "health_inspection_rating",
        ]].dropna(subset=["quarter"]).copy()
        # Group by quarter and take the latest processing date per quarter
        ratings_df = ratings_df.sort_values("processing_date").groupby("quarter").last().reset_index()
        # Add file information for tooltips
        ratings_df["file_info"] = ratings_df["processing_date"].dt.strftime("%Y-%m")
        # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
        ratings_df["quarter_label"] = ratings_df["quarter"].apply(lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) else "")
        ratings_long = ratings_df.melt(
            id_vars=["quarter", "quarter_label", "file_info"],
            value_vars=["overall_rating", "staffing_rating", "health_inspection_rating"],
            var_name="series",
            value_name="value",
        )
        series_names = {
            "overall_rating": "Overall",
            "staffing_rating": "Staffing",
            "health_inspection_rating": "Health Inspection",
        }
        ratings_long["series"] = ratings_long["series"].map(series_names)
        fig_rating = px.line(
            ratings_long,
            x="quarter_label",
            y="value",
            color="series",
            markers=True,
            title="Ratings over time (by quarter)",
            color_discrete_map={
                "Overall": "#1f77b4",
                "Staffing": "#2ca02c",
                "Health Inspection": "#ff7f0e",
            },
            custom_data=["file_info"],
        )
        # Update tooltips to include file information
        fig_rating.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Quarter: %{x}<br>Rating: %{y}<br>File: %{customdata[0]}<extra></extra>"
        )
        fig_rating.update_layout(
            title_x=0.5, 
            yaxis_title="Rating (1-5)",
            xaxis=dict(
                tickangle=45,
                nticks=len(ratings_long['quarter_label'].unique())  # Show all quarters
            )
        )
        st.plotly_chart(fig_rating, use_container_width=True)

        # Ownership change tracking
        st.markdown("---")
        st.subheader("Ownership Change History")
        
        # Get ownership change data for this facility
        ownership_changes = fac[
            fac["provider_changed_ownership_in_last_12_months"] == "Y"
        ].copy()
        
        if not ownership_changes.empty:
            ownership_changes = ownership_changes.sort_values("processing_date")
            
            # Create a simple table showing the category, value, and processing date
            change_summary = ownership_changes[[
                "processing_date", 
                "provider_changed_ownership_in_last_12_months"
            ]].copy()
            change_summary["Processing Date"] = pd.to_datetime(change_summary["processing_date"]).dt.strftime("%Y-%m-%d")
            change_summary["Category"] = "Provider Changed Ownership in Last 12 Months"
            change_summary["Value"] = change_summary["provider_changed_ownership_in_last_12_months"]
            change_summary = change_summary[["Category", "Value", "Processing Date"]].drop_duplicates()
            
            st.markdown(f"**Evidence of ownership changes found in {len(change_summary)} records:**")
            st.dataframe(change_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No evidence of ownership changes found in the available data.")

        # Basic facility information table (toggleable)
        st.markdown("---")
        st.subheader("Basic Facility Information")
        
        # Get the most recent record
        latest_record = fac.iloc[-1]
        
        # Create basic info table
        def format_rating(rating):
            """Format rating as integer if it's a number, otherwise return as string"""
            if pd.notna(rating) and str(rating).replace('.', '').isdigit():
                return int(float(rating))
            return rating
        
        def format_quarter(quarter):
            """Format quarter as Q1 2025 instead of 2025Q1"""
            if pd.notna(quarter) and len(str(quarter)) == 6:
                return f"Q{quarter[-1]} {quarter[:4]}"
            return quarter
        
        basic_info = {
            "Provider Name": latest_record.get("provider_name", ""),
            "City": latest_record.get("city", ""),
            "State": latest_record.get("state", ""),
            "County": latest_record.get("county", ""),
            "Ownership Type": latest_record.get("ownership_type", ""),
            "For Profit": latest_record.get("for_profit", ""),
            "Overall Rating": format_rating(latest_record.get("overall_rating", "")),
            "Staffing Rating": format_rating(latest_record.get("staffing_rating", "")),
            "Health Inspection Rating": format_rating(latest_record.get("health_inspection_rating", "")),
            "Matched Quarter": format_quarter(latest_record.get("quarter", "")),
            "Latest Processing Date": pd.to_datetime(latest_record.get("processing_date", "")).strftime("%Y-%m-%d") if pd.notna(latest_record.get("processing_date")) else ""
        }
        
        # Convert to DataFrame for display
        basic_df = pd.DataFrame(list(basic_info.items()), columns=["Field", "Value"])
        
        # Toggle to show/hide
        show_basic = st.checkbox("Show basic facility information", value=True)
        if show_basic:
            st.dataframe(basic_df, use_container_width=True, hide_index=True)

        # Comprehensive facility data table at the bottom
        st.markdown("---")
        st.subheader("Complete Facility Data by Quarter")
        st.caption("All available data for this facility across all quarters")
        
        # Create comprehensive table with all metrics
        comprehensive_data = fac.copy()
        
        # Add month/year column from processing_date
        comprehensive_data["Month/Year"] = pd.to_datetime(comprehensive_data["processing_date"]).dt.strftime("%Y-%m")
        
        # Add data release date (processing_date formatted)
        comprehensive_data["Data Release Date"] = pd.to_datetime(comprehensive_data["processing_date"]).dt.strftime("%Y-%m-%d")
        
        # Select and order columns for the comprehensive table
        table_columns = [
            "Month/Year",
            "Data Release Date", 
            "quarter",
            "avg_residents_per_day",
            "overall_rating",
            "staffing_rating",
            "health_inspection_rating",
            # Total nurse staffing (prioritized)
            "reported_total_nurse_hrs_per_resident_per_day",
            "case_mix_total_nurse_hrs_per_resident_per_day",
            "adjusted_total_nurse_hrs_per_resident_per_day",
            # RN staffing
            "reported_rn_hrs_per_resident_per_day",
            "case_mix_rn_hrs_per_resident_per_day",
            "adjusted_rn_hrs_per_resident_per_day",
            # LPN staffing
            "reported_lpn_hrs_per_resident_per_day",
            "case_mix_lpn_hrs_per_resident_per_day",
            "adjusted_lpn_hrs_per_resident_per_day",
            # Aide staffing
            "reported_na_hrs_per_resident_per_day",
            "case_mix_na_hrs_per_resident_per_day",
            "adjusted_na_hrs_per_resident_per_day",
            # Other staffing
            "reported_licensed_hrs_per_resident_per_day",
            "reported_pt_hrs_per_resident_per_day",
            # Weekend staffing
            "weekend_total_nurse_hrs_per_resident_per_day",
            "weekend_rn_hrs_per_resident_per_day",
            "case_mix_weekend_total_nurse_hrs_per_resident_per_day",
            "adjusted_weekend_total_nurse_hrs_per_resident_per_day",
            # Turnover metrics
            "total_nursing_staff_turnover",
            "registered_nurse_turnover",
            # Other
            "provider_changed_ownership_in_last_12_months"
        ]
        
        # Create the comprehensive table - only include columns that exist
        existing_columns = [col for col in table_columns if col in comprehensive_data.columns]
        comprehensive_table = comprehensive_data[existing_columns].copy()
        
        # Format quarter column to show Q1 2025 instead of 2025Q1
        comprehensive_table["quarter"] = comprehensive_table["quarter"].apply(
            lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) and len(str(x)) == 6 else x
        )
        
        # Create better column labels
        column_labels = {
            "Month/Year": "Month/Year",
            "Data Release Date": "Data Release Date",
            "quarter": "Matching Quarter",
            "avg_residents_per_day": "Avg Residents/Day",
            "overall_rating": "Overall Rating",
            "staffing_rating": "Staffing Rating",
            "health_inspection_rating": "Health Inspection Rating",
            "reported_na_hrs_per_resident_per_day": "Reported CNA HPRD",
            "reported_lpn_hrs_per_resident_per_day": "Reported LPN HPRD",
            "reported_rn_hrs_per_resident_per_day": "Reported RN HPRD",
            "reported_licensed_hrs_per_resident_per_day": "Reported Licensed HPRD",
            "reported_total_nurse_hrs_per_resident_per_day": "Reported Total HPRD",
            "weekend_total_nurse_hrs_per_resident_per_day": "Weekend Total HPRD",
            "weekend_rn_hrs_per_resident_per_day": "Weekend RN HPRD",
            "reported_pt_hrs_per_resident_per_day": "Reported PT HPRD",
            "case_mix_na_hrs_per_resident_per_day": "Case-Mix CNA HPRD",
            "case_mix_lpn_hrs_per_resident_per_day": "Case-Mix LPN HPRD",
            "case_mix_rn_hrs_per_resident_per_day": "Case-Mix RN HPRD",
            "case_mix_total_nurse_hrs_per_resident_per_day": "Case-Mix Total HPRD",
            "case_mix_weekend_total_nurse_hrs_per_resident_per_day": "Case-Mix Weekend Total HPRD",
            "adjusted_na_hrs_per_resident_per_day": "Adjusted CNA HPRD",
            "adjusted_lpn_hrs_per_resident_per_day": "Adjusted LPN HPRD",
            "adjusted_rn_hrs_per_resident_per_day": "Adjusted RN HPRD",
            "adjusted_total_nurse_hrs_per_resident_per_day": "Adjusted Total HPRD",
            "adjusted_weekend_total_nurse_hrs_per_resident_per_day": "Adjusted Weekend Total HPRD",
            "total_nursing_staff_turnover": "Total Nursing Staff Turnover",
            "registered_nurse_turnover": "Registered Nurse Turnover",
            "provider_changed_ownership_in_last_12_months": "Ownership Change (12mo)"
        }
        
        # Rename columns
        comprehensive_table = comprehensive_table.rename(columns=column_labels)
        
        # Sort by processing date (most recent first)
        comprehensive_table = comprehensive_table.sort_values("Data Release Date", ascending=False)
        
        # Format numeric columns to 3 decimal places
        numeric_columns = [
            "Avg Residents/Day",
            "Reported CNA HPRD", "Reported LPN HPRD", "Reported RN HPRD", "Reported Licensed HPRD", "Reported Total HPRD",
            "Weekend Total HPRD", "Weekend RN HPRD", "Reported PT HPRD",
            "Case-Mix CNA HPRD", "Case-Mix LPN HPRD", "Case-Mix RN HPRD", "Case-Mix Total HPRD", "Case-Mix Weekend Total HPRD",
            "Adjusted CNA HPRD", "Adjusted LPN HPRD", "Adjusted RN HPRD", "Adjusted Total HPRD", "Adjusted Weekend Total HPRD",
            "Total Nursing Staff Turnover", "Registered Nurse Turnover"
        ]
        
        for col in numeric_columns:
            if col in comprehensive_table.columns:
                comprehensive_table[col] = comprehensive_table[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) and x != "" else ""
                )
        
        # Display the comprehensive table
        st.dataframe(
            comprehensive_table,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Add footnote for ratings
        st.caption("*Ratings may include footnote codes (1, 2, 6, 9, 10, 12, 13, 14, 18, 19) indicating data limitations. See 'Ratings Methodology & Footnotes' above for details.")
        
        # Add summary statistics
        st.markdown("**Summary Statistics:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(comprehensive_table))
            st.metric("Quarters Covered", comprehensive_table["Matching Quarter"].nunique())
        
        with col2:
            latest_overall = comprehensive_table["Overall Rating"].iloc[0] if not comprehensive_table.empty else "N/A"
            latest_staffing = comprehensive_table["Staffing Rating"].iloc[0] if not comprehensive_table.empty else "N/A"
            st.metric("Latest Overall Rating", latest_overall)
            st.metric("Latest Staffing Rating", latest_staffing)
        
        with col3:
            latest_census = comprehensive_table["Avg Residents/Day"].iloc[0] if not comprehensive_table.empty else "N/A"
            latest_total_hprd = comprehensive_table["Reported Total HPRD"].iloc[0] if not comprehensive_table.empty else "N/A"
            st.metric("Latest Census", latest_census)
            st.metric("Latest Total HPRD", latest_total_hprd)
    
    with right:
        pass


def main() -> None:
    st.set_page_config(page_title="Provider Info Dashboard", layout="wide")
    st.title("Provider Info Facility Dashboard")
    st.caption("Staffing data is mapped to quarters based on 6-month processing delay pattern.")

    df = load_data()
    selected_ccn, _latest = facility_search(df)
    if selected_ccn:
        show_facility_page(df, str(selected_ccn))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STREAMLIT APP STARTING...")
    print("📊 Provider Info Dashboard")
    print("🌐 App will be available at: http://localhost:8502")
    print("="*60 + "\n")
    main()


