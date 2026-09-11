from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_RESULTS = Path("data/processed/anomaly_results.csv")
MODEL_COMPARISON = Path("data/processed/model_comparison.json")
RUN_SUMMARY = Path("data/processed/run_summary.json")
KB_PATH = Path("docs/knowledge_base/water_ops_guidance.md")


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_RESULTS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Standardize column references
    if "water_consumption_litres" not in df.columns and "water_consumption_liters" in df.columns:
        df["water_consumption_litres"] = df["water_consumption_liters"]
    if "occupancy" not in df.columns and "occupancy_estimate" in df.columns:
        df["occupancy"] = df["occupancy_estimate"]
    if "temperature" not in df.columns and "temperature_c" in df.columns:
        df["temperature"] = df["temperature_c"]
    if "is_anomaly" not in df.columns and "anomaly_label" in df.columns:
        df["is_anomaly"] = df["anomaly_label"]
    if "event_id" not in df.columns and "anomaly_event_id" in df.columns:
        df["event_id"] = df["anomaly_event_id"]

    return df


@st.cache_data
def load_comparison() -> dict:
    if MODEL_COMPARISON.exists():
        try:
            return json.loads(MODEL_COMPARISON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@st.cache_data
def load_summary() -> dict:
    if RUN_SUMMARY.exists():
        try:
            return json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def main() -> None:
    st.set_page_config(
        page_title="AquaGuard AI | Sustainable Campus Water Decision Support",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8fafc;
            border-left: 5px solid #0284c7;
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .badge-sdg {
            background-color: #26bde2;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-warning {
            background-color: #f59e0b;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .badge-high {
            background-color: #ef4444;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .badge-med {
            background-color: #f59e0b;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .badge-low {
            background-color: #10b981;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------
    # SECTION A: OVERVIEW & HEADER
    # ----------------------------------------------------
    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.title("💧 AquaGuard AI")
        st.subheader("AI-Based Water Consumption Anomaly Detection & Decision Support for Sustainable Campuses")
    with col_header2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<span class="badge-sdg">UN SDG 6: Clean Water</span>', unsafe_allow_html=True)
        st.caption("Campus Facility Intelligence")

    # Guardrail banner
    st.info(
        "🔬 **Important Scientific Scope & Responsible AI Notice**: "
        "AquaGuard AI analyzes **aggregate facility consumption patterns**. It detects statistical and machine learning anomalies "
        "which may indicate fixture malfunctions, irrigation schedule overlaps, tank float issues, or abnormal operational usage. "
        "**This system does NOT confirm physical pipe leaks from aggregate data alone.** "
        "All alerts serve as decision support and **require on-site human verification** before maintenance action. "
        "No autonomous shutoff is performed."
    )

    df_raw = load_data()
    comparison = load_comparison()
    summary = load_summary()

    if df_raw.empty:
        st.warning("⚠️ No processed pipeline results found at `data/processed/anomaly_results.csv`.")
        st.write("Please run `python tools/run_pipeline.py` from your terminal to generate data and model evaluation artifacts.")
        return

    # ----------------------------------------------------
    # SIDEBAR CONTROLS & FILTERS
    # ----------------------------------------------------
    st.sidebar.header("🎯 Navigation & Filters")

    # Building filter
    all_buildings = ["All Buildings"] + sorted(df_raw["building_name"].unique().tolist())
    selected_building = st.sidebar.selectbox("Select Building", all_buildings)

    # Date range filter
    min_date = df_raw["timestamp"].min().date()
    max_date = df_raw["timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range (Evaluation Window)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Risk level filter
    risk_options = ["LOW", "MEDIUM", "HIGH"]
    selected_risks = st.sidebar.multiselect("Filter by Risk Level", risk_options, default=risk_options)

    # Anomaly status filter
    view_anomaly_only = st.sidebar.checkbox("Show Flagged Anomalies Only", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Campus Buildings (6 Monitored)**:")
    for b in sorted(df_raw["building_name"].unique()):
        st.sidebar.markdown(f"- 🏢 {b}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Data Source: **Synthetic Campus Telemetry** (90 days, hourly, 6 facilities, Seed 42).")

    # Apply filters
    filtered_df = df_raw.copy()
    if selected_building != "All Buildings":
        filtered_df = filtered_df[filtered_df["building_name"] == selected_building]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_ts = pd.to_datetime(date_range[0], utc=True)
        end_ts = pd.to_datetime(date_range[1], utc=True) + pd.Timedelta(days=1)
        filtered_df = filtered_df[(filtered_df["timestamp"] >= start_ts) & (filtered_df["timestamp"] < end_ts)]

    if selected_risks:
        filtered_df = filtered_df[filtered_df["risk_tier"].isin(selected_risks)]

    if view_anomaly_only:
        filtered_df = filtered_df[filtered_df["is_anomaly"] == 1]

    # ----------------------------------------------------
    # SECTION B: KPI CARDS
    # ----------------------------------------------------
    st.markdown("### 📊 Key Performance Indicators (Telemetry & Anomaly Impact)")

    water_col = "water_consumption_litres"
    total_observed = float(filtered_df[water_col].sum())
    total_expected = float(filtered_df["expected_consumption"].sum())
    potential_excess = float(filtered_df["potential_excess_consumption"].sum())
    anomalies_count = int((filtered_df["is_anomaly"] == 1).sum())
    high_risk_count = int((filtered_df["risk_tier"] == "HIGH").sum())
    excess_pct = ((potential_excess / total_expected) * 100) if total_expected > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric(
            label="Total Water Consumed",
            value=f"{total_observed:,.0f} L",
            help="Cumulative observed meter consumption across selected scope",
        )
    with kpi2:
        st.metric(
            label="Expected Baseline",
            value=f"{total_expected:,.0f} L",
            help="Dynamic baseline from occupancy-adjusted building/hour/weekend model",
        )
    with kpi3:
        st.metric(
            label="Potential Excess Consumption",
            value=f"{potential_excess:,.0f} L",
            delta=f"+{excess_pct:.1f}% vs Expected",
            delta_color="inverse",
            help="Simulated potential avoidable consumption: max(0, observed - expected)",
        )
    with kpi4:
        st.metric(
            label="Anomalous Hours Flagged",
            value=f"{anomalies_count:,} hrs",
            help="Total hourly readings flagged as potential abnormal usage",
        )
    with kpi5:
        st.metric(
            label="High-Risk Alerts",
            value=f"{high_risk_count:,} events",
            delta=f"{high_risk_count} Urgent" if high_risk_count > 0 else "None",
            delta_color="inverse",
            help="Anomalies with elevated deviation, persistence, or overnight occurrence",
        )

    # Tabs for structured navigation
    tab_dashboard, tab_details, tab_model_eval, tab_responsible_ai = st.tabs(
        ["📈 Operations & Monitoring", "🔍 Anomaly Investigation & RAG", "🤖 Model Comparison & Baseline", "⚖️ Responsible AI & SDG 6"]
    )

    with tab_dashboard:
        # ----------------------------------------------------
        # SECTION C & D: CONSUMPTION TREND & ACTUAL VS EXPECTED
        # ----------------------------------------------------
        st.markdown("#### 📉 Consumption Trend: Observed Telemetry vs Expected Baseline")
        st.caption(
            "Visual comparison between actual observed hourly telemetry and the building-specific expected consumption baseline. "
            "Deviations illustrate potential abnormal spikes, sustained overnight usage, or gradual drift."
        )

        # Resample or group for visualization if 'All Buildings'
        if selected_building == "All Buildings":
            plot_df = (
                filtered_df.groupby("timestamp", as_index=False)
                .agg({water_col: "sum", "expected_consumption": "sum", "is_anomaly": "max"})
                .sort_values("timestamp")
            )
            plot_title = "Campus-Wide Total Hourly Consumption (L)"
        else:
            plot_df = filtered_df.sort_values("timestamp")
            plot_title = f"{selected_building}: Hourly Consumption (L)"

        st.line_chart(
            data=plot_df.set_index("timestamp")[[water_col, "expected_consumption"]],
            color=["#0284c7", "#94a3b8"],
            use_container_width=True,
        )

        # ----------------------------------------------------
        # SECTION E: BUILDING COMPARISON
        # ----------------------------------------------------
        st.markdown("#### 🏢 Building-Level Performance Comparison")
        st.caption("Fair, building-specific comparison accounting for occupancy, building function, and operational profile.")

        b_summary = (
            df_raw.groupby(["building_id", "building_name"], as_index=False)
            .agg(
                observed_litres=(water_col, "sum"),
                expected_litres=("expected_consumption", "sum"),
                potential_excess_litres=("potential_excess_consumption", "sum"),
                anomaly_hours=("is_anomaly", "sum"),
                high_risk_events=("risk_tier", lambda s: int((s == "HIGH").sum())),
            )
        )
        b_summary["deviation_pct"] = (
            (b_summary["observed_litres"] - b_summary["expected_litres"]) / b_summary["expected_litres"] * 100
        ).round(1)
        b_summary["excess_share_pct"] = (
            b_summary["potential_excess_litres"] / b_summary["potential_excess_litres"].sum() * 100
        ).round(1)

        col_table, col_bar = st.columns([3, 2])
        with col_table:
            st.dataframe(
                b_summary.rename(
                    columns={
                        "building_name": "Building",
                        "observed_litres": "Observed (L)",
                        "expected_litres": "Expected (L)",
                        "potential_excess_litres": "Potential Excess (L)",
                        "deviation_pct": "Deviation (%)",
                        "anomaly_hours": "Anomalies (hrs)",
                        "high_risk_events": "High Risk Alerts",
                    }
                ).drop(columns=["building_id"]),
                use_container_width=True,
            )
        with col_bar:
            st.bar_chart(
                data=b_summary.set_index("building_name")["potential_excess_litres"],
                color="#0284c7",
                use_container_width=True,
            )
            st.caption("Potential Excess Water (L) by Facility")

        # ----------------------------------------------------
        # SECTION F & G: ANOMALY TIMELINE & RISK DISTRIBUTION
        # ----------------------------------------------------
        col_f, col_g = st.columns([3, 2])
        with col_f:
            st.markdown("#### ⏱️ Anomaly Event Timeline")
            st.caption("Chronological distribution of abnormal consumption events across facilities.")
            
            anom_events = df_raw[df_raw["is_anomaly"] == 1].dropna(subset=["event_id"])
            if not anom_events.empty:
                event_summary = (
                    anom_events.groupby(["event_id", "building_name", "anomaly_type", "risk_tier"], as_index=False)
                    .agg(
                        start_time=("timestamp", "min"),
                        duration_hours=("timestamp", "count"),
                        peak_deviation=("percentage_deviation", "max"),
                        total_excess=("potential_excess_consumption", "sum"),
                    )
                    .sort_values("start_time", ascending=False)
                )
                st.dataframe(
                    event_summary.head(15).rename(
                        columns={
                            "event_id": "Event ID",
                            "building_name": "Building",
                            "anomaly_type": "Anomaly Type",
                            "risk_tier": "Risk Tier",
                            "start_time": "Start Time (UTC)",
                            "duration_hours": "Duration (hrs)",
                            "peak_deviation": "Peak Dev (%)",
                            "total_excess": "Excess (L)",
                        }
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No anomalous events found in the current filter range.")

        with col_g:
            st.markdown("#### 🎯 Risk Distribution")
            st.caption("Categorization of events based on magnitude, persistence, and timing.")
            risk_counts = df_raw[df_raw["is_anomaly"] == 1]["risk_tier"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
            st.bar_chart(risk_counts, color="#f59e0b", use_container_width=True)
            st.markdown(
                """
                - **HIGH**: Deviation > 70%, sustained multi-hour duration, or overnight low-occupancy anomaly.
                - **MEDIUM**: Deviation 40%–70%, moderate duration or afternoon anomaly.
                - **LOW**: Minor deviation < 40%, transient spike or early warning.
                """
            )

    with tab_details:
        # ----------------------------------------------------
        # SECTION H, I, J: DRILL-DOWN, EXPLAINABILITY & RAG
        # ----------------------------------------------------
        st.markdown("### 🔍 Anomaly Event Drill-Down & Decision Support")
        st.caption("Select any flagged anomaly to inspect transparent AI explanations, possible causes, and grounded RAG recommendations.")

        anomalies_only = df_raw[df_raw["is_anomaly"] == 1].sort_values("timestamp", ascending=False)
        if anomalies_only.empty:
            st.info("No anomalies recorded.")
        else:
            event_options = [
                f"{r['event_id']} | {r['building_name']} | {r['timestamp'].strftime('%Y-%m-%d %H:00')} | Risk: {r['risk_tier']} | {r['anomaly_type']}"
                for _, r in anomalies_only.iterrows()
            ]
            selected_event_str = st.selectbox("Select Anomaly Incident to Inspect", event_options)
            selected_idx = event_options.index(selected_event_str)
            selected_row = anomalies_only.iloc[selected_idx]

            # SECTION I: AI EXPLANATION
            st.markdown("#### 🤖 AI Transparent Explanation")
            col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
            col_exp1.metric("Observed Consumption", f"{selected_row[water_col]:,.0f} L")
            col_exp2.metric("Expected Baseline", f"{selected_row['expected_consumption']:,.0f} L")
            col_exp3.metric("Deviation", f"+{selected_row['percentage_deviation']:.1f}%")
            col_exp4.metric("Assigned Risk Tier", selected_row["risk_tier"])

            st.markdown(
                f"""
                <div class="metric-card">
                    <h5>📋 Why Was This Flagged?</h5>
                    <p>{selected_row.get('explanation', 'Deviation significantly exceeds normal expected baseline.')}</p>
                    <h5>🔍 Possible Operational Causes (Not Confirmed Leaks):</h5>
                    <p>{selected_row.get('possible_causes', 'Fixture malfunction, irrigation schedule overlap, or tank float issue.')}</p>
                    <h5>⚠️ Scientific Uncertainty & Limitation Notice:</h5>
                    <p><em>{selected_row.get('uncertainty_statement', 'Potential abnormal usage identified from aggregate meter data. Requires on-site verification.')}</em></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # SECTION J: RECOMMENDED ACTIONS (LOCAL RAG)
            st.markdown("#### 📚 Grounded Operational Recommendations (Local Knowledge Base)")
            st.caption("Evidence-grounded action protocol retrieved from local campus maintenance guidance.")

            rec_text = selected_row.get("recommendation", "Perform on-site visual check and verify meter telemetry.")
            st.markdown(
                f"""
                <div style="background-color: #ecfdf5; border-left: 5px solid #10b981; padding: 14px 18px; border-radius: 6px;">
                    <h5 style="color: #065f46; margin-top: 0;">🛠️ Prescribed Next Actions:</h5>
                    <pre style="white-space: pre-wrap; font-family: inherit; font-size: 0.95rem; color: #1e293b; background: transparent; border: none; padding: 0;">{rec_text}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # SECTION H: DETAILED TABULAR LOGS
            st.markdown("#### 📋 Incident Registry (Filtered Range)")
            display_cols = [
                "timestamp",
                "building_name",
                water_col,
                "expected_consumption",
                "percentage_deviation",
                "potential_excess_consumption",
                "anomaly_type",
                "risk_tier",
                "event_id",
            ]
            st.dataframe(
                filtered_df[display_cols].sort_values("timestamp", ascending=False),
                use_container_width=True,
            )

    with tab_model_eval:
        # ----------------------------------------------------
        # SECTION: MODEL COMPARISON & BASELINE DEMONSTRATION
        # ----------------------------------------------------
        st.markdown("### 🧪 Baseline vs Machine Learning Evaluation")
        st.markdown(
            "To prove the scientific value of AI beyond a simple fixed rule, AquaGuard AI evaluates four distinct approaches "
            "on a held-out temporal test set (last 30% of data, zero temporal leakage)."
        )

        if comparison:
            eval_rows = []
            method_names = {
                "pred_static": "Static Threshold (>35% dev)",
                "pred_statistical": "Rolling Statistical (Z-Score > 2.5)",
                "pred_ml": "Isolation Forest (ML Unsupervised)",
                "pred_hybrid": "Hybrid Ensemble (Statistical + ML)",
            }
            for k, v in comparison.items():
                cm = v.get("confusion_matrix", {})
                eval_rows.append(
                    {
                        "Model / Method": method_names.get(k, k),
                        "Precision": f"{v.get('precision', 0.0):.3f}",
                        "Recall": f"{v.get('recall', 0.0):.3f}",
                        "F1-Score": f"{v.get('f1', 0.0):.3f}",
                        "False Positive Rate": f"{v.get('false_positive_rate', 0.0):.4f}",
                        "Event Recall": f"{v.get('event_recall', 0.0):.3f}",
                        "Mean Delay (hrs)": f"{v.get('mean_detection_delay_hours', 0.0):.1f}" if not np.isnan(v.get('mean_detection_delay_hours', 0)) else "N/A",
                        "True Positives": cm.get("true_positive", 0),
                        "False Positives": cm.get("false_positive", 0),
                        "False Negatives": cm.get("false_negative", 0),
                    }
                )
            st.table(pd.DataFrame(eval_rows))

            st.markdown(
                """
                **Why ML Adds Value Beyond Fixed Rules**:
                - **Static Rules** either miss gradual drift anomalies (false negatives) or produce excessive false alarms when campus baseline shifts naturally.
                - **Rolling Statistical Methods** adapt to moving averages but struggle with non-linear multi-variate correlations (e.g., temperature vs. occupancy vs. day-of-week).
                - **Isolation Forest & Hybrid Ensembles** successfully isolate complex contextual anomalies (e.g. moderate unexpected flow during low-occupancy periods) with lower false alarm rates.
                """
            )
        else:
            st.info("Model comparison data is generating or unavailable.")

    with tab_responsible_ai:
        # ----------------------------------------------------
        # SECTION K & L: RESPONSIBLE AI & LIMITATIONS
        # ----------------------------------------------------
        st.markdown("### ⚖️ Responsible AI Framework (6 Principles)")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(
                """
                #### 1. Fairness & Equity
                Campuses house heterogeneous facilities (e.g., residential dormitories vs. academic lecture halls vs. administrative offices).
                AquaGuard AI builds **building-specific, occupancy-adjusted baseline models** so that residential hostels are not unfairly penalized
                for legitimate evening/weekend water usage.

                #### 2. Transparency & Explainability
                The system avoids black-box scoring. Every alert itemizes:
                - Exact observed consumption
                - Expected median baseline
                - Percentage deviation
                - Contextual rationale and candidate root causes

                #### 3. Mandatory Human Oversight
                AI outputs serve as **decision support for facilities technicians**.
                Maintenance staff must physically verify on-site conditions before dispatching contractors or replacing infrastructure.
                """
            )
        with col_r2:
            st.markdown(
                """
                #### 4. Data Privacy
                Telemetry is aggregated strictly at the facility/building bulk meter level.
                No individual student, staff, or room-level behavioral tracking is collected or stored.

                #### 5. Uncertainty Acknowledgment
                The system strictly adheres to scientific caution:
                **Anomaly ≠ Confirmed Leak**.
                Consumption surges can stem from legitimate campus activities (cleaning, event crowds, lab experiments).

                #### 6. Safety & Non-Autonomous Action
                AquaGuard AI is **strictly read-only and advisory**.
                The software cannot autonomously trigger physical valve shutoffs or disrupt campus water access.
                """
            )

        st.markdown("---")
        st.markdown("### 🔬 Scientific Limitations & Disclosures")
        st.markdown(
            """
            1. **Synthetic Telemetry**: The evaluation dataset is deterministically generated to model campus behavior for research, development, and hackathon demonstration. Real-world deployment will require calibration against physical sub-meter pulse loggers.
            2. **Aggregate Limitation**: Aggregate building meters cannot isolate the exact physical fixture or underground joint responsible for anomalous consumption.
            3. **Unsupervised Trade-offs**: In absence of real-world historical fault logs, unsupervised detectors may exhibit false positives during irregular academic calendars (e.g., orientation weeks, sports tournaments).
            """
        )

        st.markdown("---")
        st.markdown("### 🌍 Impact on UN SDG 6: Clean Water and Sanitation")
        st.markdown(
            f"""
            - **Target 6.4 (Water-Use Efficiency)**: By providing early detection of avoidable excess usage, campus facilities can reduce average detection lag from weeks to hours.
            - **Potential Avoidable Consumption**: In the evaluated 90-day test window, AquaGuard AI identified approximately **{summary.get('potential_excess_litres_test', 0.0):,.0f} Litres** of potential excess consumption across monitored buildings.
            - **Honest Impact Claim**: These figures represent *simulated potential indicators*, not proven historical savings from deployed hardware.
            """
        )


if __name__ == "__main__":
    main()
