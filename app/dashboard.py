"""
Customer Segmentation & Retention Dashboard
--------------------------------------------
RFM Analysis + K-Means Clustering, presented as a multi-page business
analytics application (sidebar navigation, Plotly visuals, dynamic
segment labeling, customer lookup, and downloadable reports).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "../data/customer_segments.csv"
MODEL_PATH = BASE_DIR / "../models/kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "../models/scaler.pkl"

PROJECT_AUTHOR = "Dhruv Jindal"
PROJECT_DATASET = "Online Retail II"

REQUIRED_COLS = {"CustomerID", "Recency", "Frequency", "Monetary", "Segment"}

COLORS = {
    "VIP Customers": "#7C3AED",
    "Regular Customers": "#2563EB",
    "At-Risk Customers": "#F59E0B",
    "Lost Customers": "#DC2626",
}
DEFAULT_COLOR = "#6B7280"

# Business-facing content for each segment. Keyed on the label text itself
# (not a cluster index), so it stays correct no matter how clusters are
# numbered after retraining.
SEGMENT_INSIGHTS = {
    "VIP Customers": {
        "icon": "👑",
        "characteristics": ["High Spend", "Frequent Purchases", "Recently Active"],
        "recommendation": "Offer premium/loyalty membership, early access to new products, "
                           "and dedicated support to protect this high-value relationship.",
    },
    "Regular Customers": {
        "icon": "🙂",
        "characteristics": ["Moderate Spend", "Consistent Purchases", "Steady Engagement"],
        "recommendation": "Use targeted upsell and cross-sell campaigns to grow order size "
                           "and move them toward VIP status.",
    },
    "At-Risk Customers": {
        "icon": "⚠️",
        "characteristics": ["Declining Frequency", "Increasing Recency", "Spend Slipping"],
        "recommendation": "Trigger a win-back campaign: personalized discount, check-in email, "
                           "or a limited-time incentive before they churn.",
    },
    "Lost Customers": {
        "icon": "💤",
        "characteristics": ["Long Inactive", "Low Historical Spend", "Rare Purchases"],
        "recommendation": "Low-cost reactivation only (e.g. broad email offer). Focus retention "
                           "budget on At-Risk customers instead, where ROI is higher.",
    },
}
FALLBACK_INSIGHT = {
    "icon": "❔",
    "characteristics": ["Profile not yet defined"],
    "recommendation": "Add this segment to SEGMENT_INSIGHTS to get a tailored recommendation.",
}


# --------------------------------------------------------------------------
# Data / model loading
# --------------------------------------------------------------------------

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some notebook exports save the id column as 'Customer ID' (with a space)
    instead of 'CustomerID'. Rather than relying on remembering to rename it
    at export time, normalize known column-name variants here so the app
    keeps working either way.
    """
    rename_map = {}
    for col in df.columns:
        stripped = col.strip()
        if stripped.replace(" ", "").lower() == "customerid":
            rename_map[col] = "CustomerID"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


@st.cache_data
def compute_cluster_to_label_map(rfm: pd.DataFrame, _kmeans, _scaler) -> dict:
    """
    Map raw K-Means cluster IDs -> business labels ("VIP Customers", etc.)
    by cross-referencing live model predictions on the existing dataset
    against the human-assigned 'Segment' column already in the data.

    This is computed at runtime from the current model + current data, so
    it stays correct even if the model is retrained and cluster indices
    shift around. Nothing about cluster order is hardcoded.
    """
    features = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].to_numpy())
    scaled = _scaler.transform(features)
    predicted = _kmeans.predict(scaled)

    mapping = {}
    for cluster_id in np.unique(predicted):
        mask = predicted == cluster_id
        labels_in_cluster = rfm.loc[mask, "Segment"]
        if labels_in_cluster.empty:
            mapping[int(cluster_id)] = f"Cluster {cluster_id}"
        else:
            mapping[int(cluster_id)] = labels_in_cluster.mode().iloc[0]
    return mapping


def segment_color(label: str) -> str:
    return COLORS.get(label, DEFAULT_COLOR)


def segment_insight(label: str) -> dict:
    return SEGMENT_INSIGHTS.get(label, FALLBACK_INSIGHT)


def revenue_contribution_table(rfm: pd.DataFrame) -> pd.DataFrame:
    """Revenue per segment, sorted descending, with % contribution."""
    revenue = rfm.groupby("Segment")["Monetary"].sum().reset_index()
    revenue["Revenue %"] = (revenue["Monetary"] / revenue["Monetary"].sum() * 100).round(2)
    return revenue.sort_values("Monetary", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Reusable UI components
# --------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-left: 5px solid #2563EB;
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .metric-card .label { font-size: 0.85rem; color: #6B7280; margin-bottom: 4px; }
        .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #111827; }
        .segment-card {
            border-radius: 14px;
            padding: 22px 24px;
            color: white;
            margin-bottom: 12px;
        }
        .segment-card h2 { margin: 0 0 6px 0; }
        .segment-card .sub { opacity: 0.9; font-size: 0.95rem; margin-bottom: 14px; }
        .segment-stats { display: flex; gap: 28px; flex-wrap: wrap; margin-bottom: 14px; }
        .segment-stats .stat-label { font-size: 0.8rem; opacity: 0.85; }
        .segment-stats .stat-value { font-size: 1.3rem; font-weight: 700; }
        .recommendation-box {
            background: rgba(255,255,255,0.15);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 0.92rem;
        }
        .app-footer {
            margin-top: 28px;
            padding-top: 12px;
            border-top: 1px solid #E5E7EB;
            color: #6B7280;
            font-size: 0.85rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{icon} {label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_segment_card(label: str, avg_monetary: float, avg_frequency: float, avg_recency: float):
    insight = segment_insight(label)
    color = segment_color(label)
    chars_html = "".join(f"<li>✓ {c}</li>" for c in insight["characteristics"])
    st.markdown(
        f"""
        <div class="segment-card" style="background:{color};">
            <h2>{insight['icon']} {label}</h2>
            <div class="segment-stats">
                <div><div class="stat-label">Average Revenue</div>
                     <div class="stat-value">£{avg_monetary:,.0f}</div></div>
                <div><div class="stat-label">Average Orders</div>
                     <div class="stat-value">{avg_frequency:.1f}</div></div>
                <div><div class="stat-label">Average Recency</div>
                     <div class="stat-value">{avg_recency:.0f} Days</div></div>
            </div>
            <ul style="margin:0 0 14px 18px; padding:0;">{chars_html}</ul>
            <div class="recommendation-box"><strong>Recommended Action:</strong> {insight['recommendation']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def download_button(df: pd.DataFrame, filename: str, label: str = "📥 Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render_footer():
    st.markdown(
        f"""<div class="app-footer">
        Developed by {PROJECT_AUTHOR} &nbsp;|&nbsp; Customer Segmentation &amp; Retention Analysis
        </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def page_overview(rfm: pd.DataFrame):
    st.markdown(
        """
        # 📊 Customer Segmentation Dashboard
        Analyze customer behavior using **RFM Analysis** and **K-Means Clustering**.
        """
    )

    st.markdown("## Business KPIs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Customers", f"{len(rfm):,}", "👥")
    with c2:
        metric_card("Total Revenue", f"£{rfm['Monetary'].sum():,.0f}", "💰")
    with c3:
        metric_card("Average Spend", f"£{rfm['Monetary'].mean():,.0f}", "🧾")
    with c4:
        metric_card("Average Orders", f"{rfm['Frequency'].mean():.1f}", "🔁")

    st.write("---")
    st.subheader("📋 Executive Summary")
    total = len(rfm)
    vip_count = len(rfm[rfm["Segment"] == "VIP Customers"])
    atrisk_count = len(rfm[rfm["Segment"] == "At-Risk Customers"])
    lost_count = len(rfm[rfm["Segment"] == "Lost Customers"])

    s1, s2, s3 = st.columns(3)
    s1.metric("👑 VIP Customers", vip_count, f"{vip_count / total:.1%}" if total else None, delta_color="off")
    s2.metric("⚠️ At-Risk Customers", atrisk_count, f"{atrisk_count / total:.1%}" if total else None, delta_color="off")
    s3.metric("💤 Lost Customers", lost_count, f"{lost_count / total:.1%}" if total else None, delta_color="off")

    st.write("")
    tab_revenue, tab_customers, tab_data = st.tabs(["📈 Revenue", "👥 Customers", "📄 Data"])

    with tab_revenue:
        revenue = revenue_contribution_table(rfm)

        fig = px.bar(
            revenue, x="Segment", y="Monetary", color="Segment",
            color_discrete_map=COLORS, text_auto=".2s",
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Revenue Contribution")
        st.dataframe(revenue, use_container_width=True)

        best = revenue.iloc[0]
        st.success(
            f"🏆 **Highest Revenue Segment: {best['Segment']}**  \n"
            f"Revenue Generated: £{best['Monetary']:,.0f}  \n"
            f"Contribution: {best['Revenue %']:.2f}%"
        )

    with tab_customers:
        counts = rfm["Segment"].value_counts().reset_index()
        counts.columns = ["Segment", "Customers"]
        fig = px.pie(
            counts, names="Segment", values="Customers", hole=0.5,
            color="Segment", color_discrete_map=COLORS,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric("VIP Customers", vip_count, f"{vip_count / total * 100:.1f}% of customers" if total else None, delta_color="off")
        with cc2:
            st.metric("Lost Customers", lost_count, f"{lost_count / total * 100:.1f}% of customers" if total else None, delta_color="off")

    with tab_data:
        st.subheader("Full Customer Segmentation Table")
        st.dataframe(rfm, use_container_width=True, height=320)
        download_button(rfm, "customer_segments_full.csv", "📥 Download Full Segmentation Report")

    render_footer()


def page_eda(rfm: pd.DataFrame):
    st.title("📈 Exploratory Data Analysis")
    st.caption("Distribution and relationships across Recency, Frequency, and Monetary value.")

    metric_choice = st.selectbox("Metric to inspect", ["Recency", "Frequency", "Monetary"])
    fig = px.histogram(
        rfm, x=metric_choice, color="Segment", color_discrete_map=COLORS,
        nbins=40, marginal="box", opacity=0.85,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RFM Relationship by Segment")
    fig2 = px.scatter(
        rfm, x="Recency", y="Monetary", size="Frequency", color="Segment",
        color_discrete_map=COLORS, opacity=0.75, size_max=28,
        hover_data=["CustomerID"] if "CustomerID" in rfm.columns else None,
    )
    fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Average RFM per Segment")
    profile = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
    st.dataframe(profile, use_container_width=True)

    render_footer()


def page_segments(rfm: pd.DataFrame):
    st.title("👥 Customer Segments")

    segment = st.selectbox("Select a segment to inspect", sorted(rfm["Segment"].unique()))
    seg_df = rfm[rfm["Segment"] == segment]

    render_segment_card(
        segment,
        avg_monetary=seg_df["Monetary"].mean(),
        avg_frequency=seg_df["Frequency"].mean(),
        avg_recency=seg_df["Recency"].mean(),
    )

    st.subheader("Segment Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Revenue", f"£{seg_df['Monetary'].mean():,.0f}")
    c2.metric("Average Orders", f"{seg_df['Frequency'].mean():.1f}")
    c3.metric("Average Recency", f"{seg_df['Recency'].mean():.0f} days")

    st.subheader(f"Customers in {segment} ({len(seg_df):,})")
    st.dataframe(seg_df, use_container_width=True, height=320)
    download_button(seg_df, f"{segment.replace(' ', '_').lower()}.csv", f"📥 Download {segment} List")

    st.write("---")
    st.subheader("🏆 Top Customers")
    top = seg_df.sort_values("Monetary", ascending=False).head(10)
    st.dataframe(top, use_container_width=True)

    st.write("---")
    st.subheader("🔍 Customer Lookup")
    if "CustomerID" not in rfm.columns:
        st.info("No CustomerID column found in the data — lookup is unavailable.")
        render_footer()
        return

    customer_id = st.text_input("Enter Customer ID")
    if customer_id:
        matches = rfm[rfm["CustomerID"].astype(str) == str(customer_id).strip()]
        if matches.empty:
            st.warning(f"No customer found with ID '{customer_id}'.")
        else:
            row = matches.iloc[0]
            l1, l2, l3, l4 = st.columns(4)
            with l1:
                metric_card("Segment", str(row["Segment"]), segment_insight(row["Segment"])["icon"])
            with l2:
                metric_card("Revenue", f"£{row['Monetary']:,.0f}", "💰")
            with l3:
                metric_card("Frequency", f"{row['Frequency']:.0f}", "🔁")
            with l4:
                metric_card("Recency", f"{row['Recency']:.0f} days", "⏱️")

            score = (
                row["Frequency"] * 0.4
                + row["Monetary"] / 1000 * 0.4
                + (365 - row["Recency"]) / 365 * 0.2
            )
            st.write("")
            st.progress(min(max(score / 10, 0.0), 1.0))
            st.caption(
                "Approximate Customer Value Score — a simple weighted heuristic for "
                "quick visual comparison, not an ML-derived score."
            )

    render_footer()


def page_prediction(rfm: pd.DataFrame, kmeans, scaler):
    st.title("🤖 Predict Customer Segment")
    st.caption("Enter RFM values for a customer to see their predicted segment and recommended action.")

    cluster_map = compute_cluster_to_label_map(rfm, kmeans, scaler)

    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Customer Inputs")
        recency = st.number_input("Recency (days since last order)", min_value=0, value=30)
        frequency = st.number_input("Frequency (number of orders)", min_value=0, value=5)
        monetary = st.number_input("Monetary (total spend, £)", min_value=0.0, value=500.0)
        predict_clicked = st.button("Predict Segment", type="primary")

    with right:
        st.subheader("Result")
        if predict_clicked:
            sample = np.array([[np.log1p(recency), np.log1p(frequency), np.log1p(monetary)]])
            scaled = scaler.transform(sample)
            cluster_id = int(kmeans.predict(scaled)[0])
            label = cluster_map.get(cluster_id, f"Cluster {cluster_id}")

            render_segment_card(label, avg_monetary=monetary, avg_frequency=frequency, avg_recency=recency)

            st.write("### Business Recommendation")
            st.info(segment_insight(label)["recommendation"])
        else:
            st.caption("Fill in the inputs on the left and click **Predict Segment** to see the result here.")

    with st.expander("How does this mapping stay correct after retraining?"):
        st.write(
            "Cluster ID → business label is computed at runtime by comparing the model's "
            "current predictions on the existing dataset against the human-assigned `Segment` "
            "labels already present in the data — not hardcoded to a fixed cluster index. "
            "Retrain the model and drop in a new `.pkl`, and the labels stay correct."
        )
        st.json({str(k): v for k, v in cluster_map.items()})

    render_footer()


def render_pipeline_diagram():
    diagram = """
    graph TD
        A[📄 CSV] --> B[🧹 Data Cleaning]
        B --> C[📊 EDA]
        C --> D[⚙️ Feature Engineering]
        D --> E[🧮 RFM]
        E --> F[🤖 K-Means]
        F --> G[📈 Retention Analysis]
        G --> H[🖥️ Dashboard]

        classDef stage fill:#2563EB,stroke:#1E3A8A,stroke-width:1px,color:#fff,rx:6,ry:6;
        class A,B,C,D,E,F,G,H stage;
    """
    html = f"""
    <div class="mermaid">
    {diagram}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true, theme: 'base' }});</script>
    """
    components.html(html, height=560, scrolling=True)


def page_about():
    st.title("ℹ️ About This Project")
    st.markdown(
        f"""
### Business Problem
Retail businesses often treat all customers the same, spending equal marketing effort
regardless of value or churn risk. This project segments customers using **RFM analysis**
(Recency, Frequency, Monetary value) and **K-Means clustering**, so retention spend can be
targeted where it has the most impact.

### Dataset
{PROJECT_DATASET}
        """
    )

    st.subheader("Pipeline")
    render_pipeline_diagram()

    st.markdown(
        f"""
### Tech Stack
`Python` · `pandas` · `scikit-learn` · `Streamlit` · `Plotly`

### Results
Segments identified allow differentiated retention strategy — e.g. loyalty perks for VIPs,
win-back campaigns for At-Risk customers, and reduced spend on Lost customers.

### Future Scope
- Automate periodic retraining and mapping refresh
- Add cohort/time-based retention curves
- A/B test retention campaigns per segment and feed results back into the model

---
**Author:** {PROJECT_AUTHOR}
        """
    )
    render_footer()


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------

def main():
    inject_css()

    with st.sidebar:
        st.title("📊 Customer Analytics")
        st.caption("Customer Segmentation & Retention")

        page = st.radio(
            "Navigate",
            ["📊 Dashboard", "📈 EDA", "👥 Segments", "🤖 Prediction", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            """
            **Tech Stack**
            - Python
            - Pandas
            - Scikit-Learn
            - Plotly
            - Streamlit
            """
        )
        st.divider()
        st.caption("RFM Analysis + K-Means Clustering")

    if page == "ℹ️ About":
        page_about()
        return

    try:
        rfm = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find customer data at `{DATA_PATH}`. Check the file path.")
        st.stop()

    rfm = normalize_columns(rfm)

    missing = REQUIRED_COLS - set(rfm.columns)
    if missing:
        st.error(f"The data is missing expected columns: {', '.join(sorted(missing))}")
        st.stop()

    if page == "📊 Dashboard":
        page_overview(rfm)
    elif page == "📈 EDA":
        page_eda(rfm)
    elif page == "👥 Segments":
        page_segments(rfm)
    elif page == "🤖 Prediction":
        try:
            kmeans = load_model(MODEL_PATH)
            scaler = load_model(SCALER_PATH)
        except FileNotFoundError as e:
            st.error(f"Could not load model files: {e}")
            st.stop()
        page_prediction(rfm, kmeans, scaler)


if __name__ == "__main__":
    main()