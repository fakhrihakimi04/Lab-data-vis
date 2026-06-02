import difflib
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Lab 1: Global Superstore Interactive Linked Views
# Built with Streamlit + Altair, as required by the assignment screenshot.
# Place the Kaggle CSV in the same folder as this file, or install kagglehub
# so the app can download the dataset automatically.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lab 1 - Global Superstore Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


KAGGLE_DATASET = "vivek468/superstore-dataset-final"

COMMON_CSV_NAMES = [
    "Global Superstore.csv",
    "global_superstore.csv",
    "superstore.csv",
    "Superstore.csv",
    "superstore_dataset.csv",
    "SuperStoreOrders.csv",
]

REQUIRED_FIELDS = {
    "category": ["category", "product category", "product_category"],
    "sales": ["sales", "sale", "sales amount", "amount", "revenue"],
    "quantity": ["quantity", "qty", "order quantity", "units"],
    "country": ["country", "ship country", "customer country"],
    "region": ["region", "market", "state region"],
    "order_date": ["order date", "order_date", "date", "orderdate"],
}


def normalize_name(value):
    """Normalize column names to detect close matches safely."""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def detect_column(columns, aliases):
    """Find the best matching column for a required field."""
    normalized_columns = {normalize_name(column): column for column in columns}

    for alias in aliases:
        normalized_alias = normalize_name(alias)
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    for alias in aliases:
        normalized_alias = normalize_name(alias)
        for normalized_column, original_column in normalized_columns.items():
            if normalized_alias in normalized_column or normalized_column in normalized_alias:
                return original_column

    close_matches = difflib.get_close_matches(
        normalize_name(aliases[0]),
        list(normalized_columns.keys()),
        n=1,
        cutoff=0.62,
    )
    if close_matches:
        return normalized_columns[close_matches[0]]

    return None


def detect_columns(df):
    """Map assignment fields to the actual dataset columns."""
    return {
        field: detect_column(df.columns, aliases)
        for field, aliases in REQUIRED_FIELDS.items()
    }


def find_local_csv():
    """Find a CSV file beside Lab1.py."""
    app_dir = Path(__file__).resolve().parent

    for file_name in COMMON_CSV_NAMES:
        candidate = app_dir / file_name
        if candidate.exists():
            return candidate

    csv_files = sorted(app_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    superstore_matches = [
        path for path in csv_files if "superstore" in path.name.lower()
    ]
    if superstore_matches:
        return superstore_matches[0]

    return None


def download_with_kagglehub():
    """Download the Kaggle dataset if kagglehub is installed."""
    try:
        import kagglehub
    except ImportError:
        return None, "kagglehub is not installed."

    try:
        dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        return None, f"kagglehub could not download the dataset: {exc}"

    csv_files = sorted(dataset_path.rglob("*.csv"))
    if not csv_files:
        return None, f"No CSV file was found inside {dataset_path}."

    return csv_files[0], None


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load, inspect, clean, and enrich the Global Superstore data."""
    csv_path = find_local_csv()
    source_note = "local file beside Lab1.py"

    if csv_path is None:
        csv_path, download_error = download_with_kagglehub()
        source_note = "downloaded with kagglehub"
        if csv_path is None:
            return None, None, None, (
                "No dataset CSV was found beside Lab1.py, and automatic Kaggle "
                f"download was not available. {download_error} "
                "Place the Global Superstore CSV beside Lab1.py or install "
                "kagglehub and configure Kaggle access."
            )

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")
    except Exception as exc:
        return None, csv_path, None, f"Could not read the dataset file: {exc}"

    detected = detect_columns(df)
    missing = [field for field, column in detected.items() if column is None]
    if missing:
        return None, csv_path, detected, (
            "The dataset was loaded, but these required columns could not be "
            f"detected automatically: {', '.join(missing)}."
        )

    cleaned = df.copy()
    cleaned[detected["sales"]] = pd.to_numeric(
        cleaned[detected["sales"]], errors="coerce"
    )
    cleaned[detected["quantity"]] = pd.to_numeric(
        cleaned[detected["quantity"]], errors="coerce"
    )
    cleaned[detected["order_date"]] = pd.to_datetime(
        cleaned[detected["order_date"]], errors="coerce"
    )

    cleaned = cleaned.dropna(
        subset=[
            detected["category"],
            detected["sales"],
            detected["quantity"],
            detected["country"],
            detected["region"],
            detected["order_date"],
        ]
    )

    cleaned["Category"] = cleaned[detected["category"]].astype(str)
    cleaned["Sales"] = cleaned[detected["sales"]].astype(float)
    cleaned["Quantity"] = cleaned[detected["quantity"]].astype(float)
    cleaned["Country"] = cleaned[detected["country"]].astype(str)
    cleaned["Region"] = cleaned[detected["region"]].astype(str)
    cleaned["Order Date"] = cleaned[detected["order_date"]]
    cleaned["Month"] = cleaned["Order Date"].dt.to_period("M").dt.to_timestamp()
    cleaned["Quarter"] = cleaned["Order Date"].dt.to_period("Q").dt.to_timestamp()
    cleaned["Year"] = cleaned["Order Date"].dt.to_period("Y").dt.to_timestamp()

    quantity_for_price = cleaned["Quantity"].where(cleaned["Quantity"] != 0)
    cleaned["Price"] = (cleaned["Sales"] / quantity_for_price).fillna(cleaned["Sales"])
    cleaned["Dataset Source"] = source_note

    return cleaned, csv_path, detected, None


def base_chart(data):
    """Common Altair chart defaults."""
    return alt.Chart(data).configure_axis(
        labelFontSize=12,
        titleFontSize=13,
    ).configure_title(
        fontSize=16,
        anchor="start",
    )


def sales_by_category(data):
    return (
        data.groupby("Category", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )


def sales_by_country(data):
    return (
        data.groupby("Country", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )


def render_dataset_info(df, csv_path, detected):
    with st.expander("Dataset inspection and detected columns", expanded=False):
        st.write(f"Dataset file: `{csv_path}`")
        st.write(f"Dataset source: `{df['Dataset Source'].iloc[0]}`")
        detected_table = pd.DataFrame(
            {
                "Required Field": list(detected.keys()),
                "Detected Dataset Column": list(detected.values()),
            }
        )
        st.dataframe(detected_table, use_container_width=True)
        st.write(f"Rows after cleaning: `{len(df):,}`")


def render_basic_analysis(df):
    st.header("Basic Analysis")

    categories = sorted(df["Category"].unique())
    min_sales = float(df["Sales"].min())
    max_sales = float(df["Sales"].max())

    filter_col_1, filter_col_2 = st.columns([2, 1])
    with filter_col_1:
        selected_sales = st.slider(
            "Sales Amount Slider Filter",
            min_value=min_sales,
            max_value=max_sales,
            value=(min_sales, max_sales),
            step=max((max_sales - min_sales) / 100, 1.0),
        )
    with filter_col_2:
        selected_category = st.selectbox(
            "Product Category Dropdown Filter",
            ["All Categories"] + categories,
        )

    filtered = df[
        (df["Sales"] >= selected_sales[0])
        & (df["Sales"] <= selected_sales[1])
    ]
    if selected_category != "All Categories":
        filtered = filtered[filtered["Category"] == selected_category]

    if filtered.empty:
        st.warning("No records match the selected filters.")
        return

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Total Sales", f"${filtered['Sales'].sum():,.2f}")
    metric_2.metric("Total Quantity", f"{filtered['Quantity'].sum():,.0f}")
    metric_3.metric("Filtered Records", f"{len(filtered):,}")

    category_sales = sales_by_category(filtered)
    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        bar_chart = alt.Chart(category_sales).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Category:N", sort="-y", title="Category"),
            y=alt.Y("Sales:Q", title="Sales"),
            color=alt.Color("Category:N", legend=None),
            tooltip=["Category:N", alt.Tooltip("Sales:Q", format=",.2f")],
        ).properties(title="Sales by Category Bar Chart", height=360)
        st.altair_chart(bar_chart, use_container_width=True)

    with chart_col_2:
        pie_chart = alt.Chart(category_sales).mark_arc(innerRadius=60).encode(
            theta=alt.Theta("Sales:Q"),
            color=alt.Color("Category:N", title="Category"),
            tooltip=["Category:N", alt.Tooltip("Sales:Q", format=",.2f")],
        ).properties(title="Sales by Category Pie Chart", height=360)
        st.altair_chart(pie_chart, use_container_width=True)


def render_page_a(df):
    st.header("Page A: Bar Chart + Scatter Plot")
    st.caption("Click a category bar to filter the scatter plot.")

    category_sales = sales_by_category(df)
    category_selection = alt.selection_point(fields=["Category"], empty="all")

    bar_chart = alt.Chart(category_sales).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Category:N", sort="-y", title="Product Category"),
        y=alt.Y("Sales:Q", title="Total Sales"),
        color=alt.condition(
            category_selection,
            alt.Color("Category:N", legend=None),
            alt.value("#d1d5db"),
        ),
        tooltip=["Category:N", alt.Tooltip("Sales:Q", format=",.2f")],
    ).add_params(category_selection).properties(
        title="Sales by Category",
        height=360,
    )

    scatter_plot = alt.Chart(df).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("Price:Q", title="Price (Sales / Quantity)"),
        y=alt.Y("Quantity:Q", title="Quantity"),
        color=alt.Color("Category:N", title="Product Category"),
        tooltip=[
            "Category:N",
            "Country:N",
            "Region:N",
            alt.Tooltip("Price:Q", format=",.2f"),
            alt.Tooltip("Sales:Q", format=",.2f"),
            alt.Tooltip("Quantity:Q", format=",.0f"),
        ],
    ).transform_filter(category_selection).properties(
        title="Price vs. Quantity for Selected Category",
        height=360,
    )

    st.altair_chart(alt.hconcat(bar_chart, scatter_plot).resolve_scale(color="independent"), use_container_width=True)


def render_page_b(df):
    st.header("Page B: Line Chart + Histogram")

    period_map = {
        "Monthly": "Month",
        "Quarterly": "Quarter",
        "Yearly": "Year",
    }
    selected_period = st.selectbox("Select time period for the line chart", list(period_map.keys()))
    period_column = period_map[selected_period]

    trend = (
        df.groupby(period_column, as_index=False)["Sales"]
        .sum()
        .sort_values(period_column)
    )

    time_selection = alt.selection_interval(encodings=["x"])

    line_chart = alt.Chart(trend).mark_line(point=True).encode(
        x=alt.X(f"{period_column}:T", title=selected_period),
        y=alt.Y("Sales:Q", title="Sales"),
        tooltip=[
            alt.Tooltip(f"{period_column}:T", title="Period"),
            alt.Tooltip("Sales:Q", format=",.2f"),
        ],
    ).add_params(time_selection).properties(
        title="Sales Trend Over Time",
        height=340,
    )

    histogram = alt.Chart(df).mark_bar().encode(
        x=alt.X("Quantity:Q", bin=alt.Bin(maxbins=20), title="Quantity"),
        y=alt.Y("count():Q", title="Number of Orders"),
        color=alt.value("#60a5fa"),
        tooltip=[
            alt.Tooltip("Quantity:Q", bin=True, title="Quantity Bin"),
            alt.Tooltip("count():Q", title="Orders"),
        ],
    ).transform_filter(time_selection).properties(
        title="Quantity Distribution for Selected Time Period",
        height=300,
    )

    st.altair_chart(alt.vconcat(line_chart, histogram), use_container_width=True)


def render_page_c(df):
    st.header("Page C: Pie Chart + Bar Chart")
    st.caption("Click a pie slice to filter the country or region bar chart.")

    category_sales = sales_by_category(df)
    location_level = st.radio("Bar chart location level", ["Country", "Region"], horizontal=True)
    category_selection = alt.selection_point(fields=["Category"], empty="all")

    pie_chart = alt.Chart(category_sales).mark_arc(innerRadius=60).encode(
        theta=alt.Theta("Sales:Q"),
        color=alt.Color("Category:N", title="Category"),
        opacity=alt.condition(category_selection, alt.value(1), alt.value(0.35)),
        tooltip=["Category:N", alt.Tooltip("Sales:Q", format=",.2f")],
    ).add_params(category_selection).properties(
        title="Sales Distribution by Category",
        height=360,
    )

    location_sales = (
        df.groupby(["Category", location_level], as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )

    bar_chart = alt.Chart(location_sales).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(f"{location_level}:N", sort="-y", title=location_level),
        y=alt.Y("Sales:Q", title="Sales"),
        color=alt.Color(f"{location_level}:N", legend=None),
        tooltip=[
            "Category:N",
            f"{location_level}:N",
            alt.Tooltip("Sales:Q", format=",.2f"),
        ],
    ).transform_filter(category_selection).properties(
        title=f"Sales by {location_level} for Selected Category",
        height=360,
    )

    st.altair_chart(alt.hconcat(pie_chart, bar_chart).resolve_scale(color="independent"), use_container_width=True)


def render_page_d(df):
    st.header("Page D: Map View + Scatter Plot")
    st.caption("Click a country bar in the map-style country view to filter the scatter plot.")

    country_sales = sales_by_country(df).head(30)
    country_selection = alt.selection_point(fields=["Country"], empty="all")

    map_view = alt.Chart(country_sales).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("Country:N", sort="-x", title="Country"),
        x=alt.X("Sales:Q", title="Sales"),
        color=alt.condition(
            country_selection,
            alt.Color("Sales:Q", scale=alt.Scale(scheme="tealblues"), legend=alt.Legend(title="Sales")),
            alt.value("#d1d5db"),
        ),
        tooltip=["Country:N", alt.Tooltip("Sales:Q", format=",.2f")],
    ).add_params(country_selection).properties(
        title="Map View: Sales by Country",
        height=520,
    )

    scatter_plot = alt.Chart(df).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("Sales:Q", title="Sales"),
        y=alt.Y("Quantity:Q", title="Quantity"),
        color=alt.Color("Category:N", title="Category"),
        tooltip=[
            "Country:N",
            "Region:N",
            "Category:N",
            alt.Tooltip("Sales:Q", format=",.2f"),
            alt.Tooltip("Quantity:Q", format=",.0f"),
        ],
    ).transform_filter(country_selection).properties(
        title="Sales vs. Quantity for Selected Country",
        height=520,
    )

    st.altair_chart(alt.hconcat(map_view, scatter_plot).resolve_scale(color="independent"), use_container_width=True)


def main():
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            div[data-testid="stMetric"] {
                background: #f8fafc;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 14px 16px;
            }
            section[data-testid="stSidebar"] {
                background: #f8fafc;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df, csv_path, detected, error_message = load_dataset()
    if error_message:
        st.title("Lab 1: Global Superstore Dashboard")
        st.error(error_message)
        st.info(
            "Install commands: `pip install streamlit pandas altair kagglehub`. "
            "Then run: `streamlit run Lab1.py`."
        )
        if detected:
            st.write("Detected columns so far:")
            st.json(detected)
        st.stop()

    with st.sidebar:
        st.title("Choose an Example")
        page = st.selectbox(
            "Dashboard page",
            [
                "Basic Analysis",
                "Bar Chart + Scatter Plot",
                "Line Chart + Histogram",
                "Pie Chart + Bar Chart",
                "Map View + Scatter Plot",
            ],
        )
        st.divider()
        st.caption("Dataset: Global Superstore from Kaggle")

    st.title("Linked Views: Global Superstore Data Visualization")
    render_dataset_info(df, csv_path, detected)

    if page == "Basic Analysis":
        render_basic_analysis(df)
    elif page == "Bar Chart + Scatter Plot":
        render_page_a(df)
    elif page == "Line Chart + Histogram":
        render_page_b(df)
    elif page == "Pie Chart + Bar Chart":
        render_page_c(df)
    elif page == "Map View + Scatter Plot":
        render_page_d(df)


if __name__ == "__main__":
    main()
