import streamlit as st
import pandas as pd

# Data for types and countries
top_ten_types = [
    "Retail",
    "Restaurant",
    "Wholesale",
    "Other",
    "Fast Food",
    "Service",
    "Trader",
    "Food Stall",
    "Cafe",
    "Coffee Shop",
]
top_ten_countries = ["PH", "IN", "MY", "MX", "NG", "US", "KE", "ZA", "ZW", "CO"]
start_years = [2022, 2023, 2024]
start_months = [i for i in range(1, 13)]
num_months = [i for i in range(1, 31)]

# Streamlit app
st.title("Retention P1M, Revenue Breakdown - P1M, P1Y")
#st.subheader("Retention Data Dashboard - PM1")
st.markdown('<p style="color:#FF6347; font-size:24px;">Retention Data Dashboard - PM1</p>', unsafe_allow_html=True)

st.subheader("Business Type and Country Selector")

# Dropdown for selecting country
COUNTRY = st.selectbox(
    "Select a country:",
    options=["ALL"] + top_ten_countries,
    index=top_ten_countries.index("IN"),
)

# Dropdown for selecting business type
BTYPE = st.selectbox("Select a business type:", options=["ALL"] + top_ten_types)

START_MONTH = st.selectbox("Select a start month:", options=start_months)
START_YEAR = st.selectbox("Select a start year:", options=start_years)
NUM_MONTHS = st.selectbox("Select number of  months:", options=num_months)


# Display selected options
st.write(f"Selected Country: **{COUNTRY}**")
st.write(f"Selected Business Type: **{BTYPE}**")
from pathlib import Path

current_file_path = Path(__file__)

# df = pd.read_parquet(r"C:\Users\mailv\projects\zobaze\processed_data\2020-01-01_TO_2024-07-31_2.parquet")
# df = pd.read_parquet(Path(current_file_path.parent,"processed_data","2020-01-01_TO_2024-07-31_2.parquet"))
df = pd.read_parquet("2020-01-01_TO_2024-07-31_2.parquet")

df["num_months_current_sub_to_first_sub"] = df["current_payment_date"].dt.to_period(
    "M"
).astype(int) - df["first_paid_td"].dt.to_period("M").astype(int)
df["first_paid_td_year"] = df["first_paid_td"].dt.year
df["first_paid_td_month"] = df["first_paid_td"].dt.month
df["first_paid_td_month_abbr"] = df["first_paid_td"].dt.strftime("%b")

df = df[
    [
        "bus_id",
        "buyer_country",
        "type",
        "base_plan_id",
        "revenue",
        "first_paid_td",
        "first_paid_td_month",
        "first_paid_td_month_abbr",
        "first_paid_td_year",
        "current_payment_date",
        "num_months_current_sub_to_first_sub",
    ]
]  # 96603
df = df[df["num_months_current_sub_to_first_sub"] >= 0]  # 92710
# Estimate null base_plan_id based on revenue
df.loc[(df["revenue"] <= 500) & (df["base_plan_id"].isnull()), "base_plan_id"] = "p1m"
df.loc[(df["revenue"] > 500) & (df["base_plan_id"].isnull()), "base_plan_id"] = "p1y"

df = df[df["base_plan_id"] == "p1m"]  # 83767, 65056
if COUNTRY != "ALL":
    df = df[df["buyer_country"] == COUNTRY]
if BTYPE != "ALL":
    df = df[df["type"] == BTYPE]

from dateutil.relativedelta import relativedelta
from datetime import datetime


def get_next_month_year_pairs(start_month, start_year, n):
    # Initialize the starting date
    start_date = datetime(start_year, start_month, 1)

    # Generate the next N month-year pairs
    result = [
        (
            (start_date + relativedelta(months=i)).month,
            (start_date + relativedelta(months=i)).year,
        )
        for i in range(n)
    ]

    return result


def get_p1m_counts(df, start_month: int, start_year: int, num_months: int, offset: int):
    out_dict = {}
    df1 = df[
        (df["first_paid_td_month"] == start_month)
        & (df["first_paid_td_year"] == start_year)
    ]
    for i in range(num_months):
        # df_temp = df[df["num_months_current_sub_to_first_sub"]==i].groupby(['buyer_country', 'type'])["bus_id"].count().reset_index()
        # out_dict.append(df_temp)
        df_temp = df1[df1["num_months_current_sub_to_first_sub"] == i]
        out_dict[f"M{i + offset}"] = df_temp.shape[0]
    return out_dict


import calendar


def get_p1m_counts_natrix(df, start_month: int, start_year: int, num_months: int = 12):
    out_dict = {}
    next_months_years = get_next_month_year_pairs(start_month, start_year, num_months)
    for pos, (month, year) in enumerate(next_months_years):
        out = get_p1m_counts(df, month, year, num_months - pos, pos)
        key = f"{calendar.month_abbr[month]}-{str(year)[2:]}"
        out_dict[key] = out
    df = pd.DataFrame.from_dict(out_dict, orient="index").fillna(0)
    return df


final_counts = get_p1m_counts_natrix(df, START_MONTH, START_YEAR, NUM_MONTHS)


# Function to calculate percentages
def row_to_percentages(row):
    first_non_zero = next((x for x in row if x != 0), None)
    if first_non_zero is not None:
        return row / first_non_zero * 100
    return row


# Apply the function to each row
final_percen = final_counts.apply(row_to_percentages, axis=1)

# Format percentages for display
final_percen = final_percen.round(2)  # Round to 2 decimal places



# Display table using st.dataframe
st.subheader("Retention Numbers")
st.dataframe(final_counts)

# Display table using st.dataframe
st.subheader("Retention Percentage")
st.dataframe(final_percen)

######################################## REVENUE SECTION ###############

# Data for types and countries
top_ten_types = [
    "Retail",
    "Restaurant",
    "Wholesale",
    "Other",
    "Fast Food",
    "Service",
    "Trader",
    "Food Stall",
    "Cafe",
    "Coffee Shop",
]
top_ten_countries = ["PH", "IN", "MY", "MX", "NG", "US", "KE", "ZA", "ZW", "CO"]
start_years = [2022, 2023, 2024]
start_months = [i for i in range(1, 13)]
num_months = [i for i in range(1, 31)]

# Streamlit app
#st.subheader("Revenue Dashboard - P1M, P1Y")
st.markdown('<p style="color:#FF6347; font-size:24px;">Revenue Dashboard - P1M, P1Y</p>', unsafe_allow_html=True)

st.subheader("Business Type and Country Selector")

# Dropdown for selecting country
COUNTRY = st.selectbox(
    "Select a country for revenue:",
    options=["ALL"] + top_ten_countries,
    index=top_ten_countries.index("IN"))

# Dropdown for selecting business type
BTYPE = st.selectbox("Select a business type for revenue:", options=["ALL"] + top_ten_types
)

START_MONTH = st.selectbox("Select a start month for revenue:", options=start_months
)
START_YEAR = st.selectbox("Select a start year for revenue:", options=start_years
)
NUM_MONTHS = st.selectbox("Select number of  months for revenue:", options=num_months
)
df2 = pd.read_parquet("2020-01-01_TO_2024-07-31_2.parquet")

if COUNTRY != "ALL":
    df2 = df2[df2["buyer_country"] == COUNTRY]
if BTYPE != "ALL":
    df2 = df2[df2["type"] == BTYPE]

def prep_df(df):
    df["num_months_current_sub_to_first_sub"] = df[
                                                    "current_payment_date"
                                                ].dt.to_period("M").astype(int) - df["first_paid_td"].dt.to_period(
        "M"
    ).astype(
        int
    )
    df['first_paid_td_year'] = df['first_paid_td'].dt.year
    df['first_paid_td_month'] = df['first_paid_td'].dt.month
    df['first_paid_td_month_abbr'] = df['first_paid_td'].dt.strftime('%b')

    df = df[["bus_id", "buyer_country", "type", "base_plan_id", "revenue", "first_paid_td", "first_paid_td_month",
             "first_paid_td_month_abbr", "first_paid_td_year",
             "current_payment_date", "num_months_current_sub_to_first_sub"]]  # 96603
    df = df[df["num_months_current_sub_to_first_sub"] >= 0]  # 92710
    # Estimate null base_plan_id based on revenue
    df.loc[(df['revenue'] <= 500) & (df['base_plan_id'].isnull()), 'base_plan_id'] = 'p1m'
    df.loc[(df['revenue'] > 500) & (df['base_plan_id'].isnull()), 'base_plan_id'] = 'p1y'
    return df


def get_revenue_type(row):
    if row["base_plan_id"] == "p1m":
        if row["num_months_current_sub_to_first_sub"] == 0:
            return "New"
        elif row["current_prior_diff_num_months"] in (0, 1):
            return "Recurring"
        elif row["current_prior_diff_num_months"] > 1:
            return "Resureccted"
        else:
            return "Unknown"
    elif row["base_plan_id"] == "p1y":
        if row["num_months_current_sub_to_first_sub"] == 0:
            return "New"
        elif row["current_prior_diff_num_months"] > 0 and row["current_prior_diff_num_months"] <= 13:
            return "Recurring"
        elif row["current_prior_diff_num_months"] > 13:
            return "Resureccted"
        else:
            return "Unknown"
    else:
        return "Unknown"


def prep_for_revenue(df):
    df["current_payment_date_month"] = df["current_payment_date"].dt.month
    df["current_payment_date_year"] = df["current_payment_date"].dt.year

    df = df.sort_values(by=["bus_id", "current_payment_date"])
    df["prior_payment_date"] = df.groupby("bus_id")["current_payment_date"].shift(1)
    df["current_prior_diff_num_months"] = df[
                                              "current_payment_date"
                                          ].dt.to_period("M").astype(int) - df["prior_payment_date"].dt.to_period(
        "M"
    ).astype(
        int
    )
    df["revenue_type"] = df.apply(get_revenue_type, axis=1)

    return df
def revenue_final(df_revenue,start_month,start_year,num_months):
    out_dict = {}
    for month, year in get_next_month_year_pairs(start_month, start_year, num_months):
        key = f"{calendar.month_abbr[month]}-{str(year)[2:]}"
        df_temp = df_revenue[
            (df_revenue["current_payment_date_month"] == month) & (df_revenue["current_payment_date_year"] == year)][
            ["revenue", "base_plan_id", "revenue_type"]]
        df_temp = df_temp.groupby(['revenue_type', 'base_plan_id'])['revenue'].sum().unstack(fill_value=0)
        out_dict[key] = df_temp.to_dict()

    # reformat
    # Flatten the dictionary into a list of records
    records = []
    for month, plans in out_dict.items():
        for plan, metrics in plans.items():
            for metric, value in metrics.items():
                records.append({"month": month, "plan": plan, "metric": metric, "value": value})

    # Create a DataFrame
    df_final = pd.DataFrame(records)

    # Pivot the DataFrame and create a multi-level index
    df_final = df_final.pivot(index=["plan", "metric"], columns="month", values="value")
    return df_final
df2 = prep_df(df2)
df2 = prep_for_revenue(df2)
df_final = revenue_final(df2,START_MONTH,START_YEAR,NUM_MONTHS)


# Display table using st.dataframe
st.subheader("Revenue Breakdown")
st.dataframe(df_final)

