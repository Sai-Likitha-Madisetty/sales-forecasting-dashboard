import plotly.express as px

import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide"
)

st.title("End-to-End Sales Forecasting & Demand Intelligence System")

st.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)



import pandas as pd

df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True
)
df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    dayfirst=True
)
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    # -------------------------------
    # KPI Cards
    # -------------------------------

    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer ID"].nunique()
    avg_sales = df["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Total Orders", total_orders)
    c3.metric("Customers", total_customers)
    c4.metric("Average Sale", f"${avg_sales:.2f}")

    st.markdown("---")    
    
    yearly_sales = (
        df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig_year = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        title="Total Sales by Year",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig_year,
        use_container_width=True
    )
    
    
    monthly_sales = (
        df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Order Date"] = (
        monthly_sales["Order Date"]
        .astype(str)
    )

    fig_month = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )
    
    
    
    st.subheader("Sales by Region & Category")

    region = st.selectbox(
        "Select Region",
        ["All"] + sorted(df["Region"].unique().tolist())
    )

    category = st.selectbox(
        "Select Category",
        ["All"] + sorted(df["Category"].unique().tolist())
    )

    filtered_df = df.copy()

    if region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == region
        ]

    if category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == category
        ]

    fig = px.bar(
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index(),

        x="Category",
        y="Sales",
        color="Category",
        title="Sales by Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
elif page == "Forecast Explorer":

    st.subheader("Sales Forecast Explorer")

    forecast_type = st.selectbox(
        "Forecast Type",
        ["Category", "Region"],
        key="forecast_type"
    )

    if forecast_type == "Category":

        selection = st.selectbox(
            "Select Category",
            ["Furniture", "Technology", "Office Supplies"],
            key="category_select"
        )

    else:

        selection = st.selectbox(
            "Select Region",
            ["West", "East"],
            key="region_select"
        )

    months = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    # Read forecast files
    overall_forecast = pd.read_csv("data/overall_forecast.csv")

    overall_forecast = overall_forecast.rename(
        columns={
            "Order Date": "ds",
            "Forecast": "yhat",
            "Actual": "y"
        }
    )
    furniture_forecast = pd.read_csv("data/furniture_forecast.csv")
    technology_forecast = pd.read_csv("data/technology_forecast.csv")
    office_forecast = pd.read_csv("data/office_forecast.csv")
    west_forecast = pd.read_csv("data/west_forecast.csv")
    east_forecast = pd.read_csv("data/east_forecast.csv")

    forecast_files = [
        overall_forecast,
        furniture_forecast,
        technology_forecast,
        office_forecast,
        west_forecast,
        east_forecast
    ]

    for file in forecast_files:
        file["ds"] = pd.to_datetime(file["ds"])

    if selection == "Furniture":
        forecast_df = furniture_forecast

    elif selection == "Technology":
        forecast_df = technology_forecast

    elif selection == "Office Supplies":
        forecast_df = office_forecast

    elif selection == "West":
        forecast_df = west_forecast

    elif selection == "East":
        forecast_df = east_forecast

    else:
        forecast_df = overall_forecast

    forecast_df = forecast_df.head(months)

    fig = px.line(
        forecast_df,
        x="ds",
        y="yhat",
        markers=True,
        title=f"{selection} Sales Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Values")

    st.dataframe(
        forecast_df.rename(
            columns={
                "ds": "Forecast Date",
                "yhat": "Predicted Sales"
            }
        ),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", "14,763.81")
    c2.metric("RMSE", "18,337.41")
    c3.metric("MAPE", "14.48%")

    st.success("Best Model Selected: XGBoost Regressor")


elif page == "Anomaly Report":

    st.header("Anomaly Report")

    anomaly_df = pd.read_csv(
        "data/anomaly_report.csv"
    )

    anomaly_df["Order Date"] = pd.to_datetime(
        anomaly_df["Order Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    st.subheader("Sales Anomaly Detection")

    fig = px.line(
        anomaly_df,
        x="Order Date",
        y="Sales",
        title="Sales with Detected Anomalies"
    )

    anomaly_points = anomaly_df[
        anomaly_df["Isolation_Anomaly"].astype(str).str.upper() == "TRUE"
    ]

    fig.add_scatter(
        x=anomaly_points["Order Date"],
        y=anomaly_points["Sales"],
        mode="markers",
        name="Anomalies"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Detected Anomaly Dates"
    )

    anomaly_table = anomaly_points[
        ["Order Date", "Sales"]
    ]

    st.dataframe(
        anomaly_table,
        use_container_width=True
    )
    
    
elif page == "Product Demand Segments":

    st.header("Product Demand Segmentation")

    cluster_df = pd.read_csv(
        "data/cluster_summary.csv"
    )

    st.subheader(
        "Demand Segment Distribution"
    )

    segment_count = (
        cluster_df["Demand_Segment"]
        .value_counts()
        .reset_index()
    )

    segment_count.columns = [
        "Demand_Segment",
        "Count"
    ]

    fig = px.bar(
        segment_count,
        x="Demand_Segment",
        y="Count",
        color="Demand_Segment",
        title="Products per Demand Segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Sub-Category Demand Segments"
    )

    st.dataframe(
        cluster_df,
        use_container_width=True
    )