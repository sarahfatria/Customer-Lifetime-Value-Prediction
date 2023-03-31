import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter, GammaGammaFitter, plotting
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

df = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df.head()
df.info()
df.describe().T
df = (
    df[~df["Invoice"].str.startswith("C", na=False)]
    .query("Quantity > 1 and Price > 0")
    .assign(TotalPrice=lambda x: x["Quantity"] * x["Price"])
)

def outlier_threshold(dataframe, variable):
    q1, q3 = dataframe[variable].quantile([0.01, 0.99])
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit

df["Customer ID"] = df["Customer ID"].astype(int)

def calculate_cltv(cltv_df, bgf, ggf, time):
    bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
    ggf.fit(cltv_df["frequency"], cltv_df["avg_monetary"])
    return ggf.customer_lifetime_value(
        bgf,
        cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"],
        cltv_df["avg_monetary"],
        time=time,
        discount_rate=0.01,
        freq="W",
    )

cltv_1_month = calculate_cltv(uk_cltv_df, bgf, ggf, 1)
cltv_6_month = calculate_cltv(uk_cltv_df, bgf, ggf, 6)
cltv_12_month = calculate_cltv(uk_cltv_df, bgf, ggf, 12)

bins = [0, 1000, 5000, 10000, np.inf]
labels = ["D", "C", "B", "A"]
uk_cltv_df["segment"] = pd.cut(uk_cltv_df["cltv_6_month"], bins=bins, labels=labels)

uk_cltv_df.groupby("segment").agg({
"cltv_6_month": ["mean", "sum", "count"],
"exp_sales_6_month": "mean",
"exp_average_value": "mean",
"expected_order": "mean"
})
