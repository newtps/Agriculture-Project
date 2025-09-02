import pandas as pd
import numpy as np
from meteostat import Point, Monthly
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import requests
from math import isnan

optimum_ranges = {
    "Jowar Kharif": {"tmax":(30, 35), "tmin":(20, 25),"rain":(600, 1000),"sunshine":(6,8)},
    "Jowar Rabi": {"tmax":(25,32), "tmin":(15,20 ), "rain":(400, 700), "sunshine":(5, 7)},
    "Wheat Rabi": {"tmax":(20, 30), "tmin":(10, 20), "rain":(300, 500), "sunshine":(6, 8)},
    "Groundnut Kharif": {"tmax":(28, 34), "tmin":(20, 24), "rain":(500, 1000), "sunshine":(7,9)},
    "Gram Rabi": {"tmax":(20, 30), "tmin":(10,20), "rain":(250, 450), "sunshine": (6, 8)},
    "Safflower Rabi": {"tmax":(22, 30), "tmin":(8, 18), "rain":(200, 400), "sunshine":(6, 9)},
    "Bajra Kharif": {"tmax":(30, 38), "tmin":(22, 28), "rain":(400, 750), "sunshine":(7, 9)},
    "Sugarcane": {"tmax":(25, 35), "tmin":(20, 25), "rain":(1000, 1500), "sunshine":(6, 8)}
}

crop_season_mapping = {
    "Jowar Kharif":[6, 7, 8, 9, 10],
    "Jowar Rabi":[9, 10, 11, 12, 1, 2, 3],
    "Wheat Rabi":[10, 11, 12, 1, 2, 3],
    "Groundnut Kharif":[6, 7, 8, 9, 10],
    "Gram Rabi":[10, 11, 12, 1, 2, 3],
    "Safflower Rabi":[10, 11, 12, 1, 2, 3],
    "Bajra Kharif":[6, 7, 8, 9, 10],
    "Sugarcane":list(range(1, 13))
}

yields_data = {
    "Jowar Kharif":[0.62, 1.13, 0.31],
    "Jowar Rabi":[0.72, 0.83, 0.93],
    "Wheat Rabi":[2.09, 2.10, 2.44],
    "Groundnut Kharif":[1.15, 0.91, 1.04],
    "Gram Rabi":[0.85, 1.06, 1.10],
    "Safflower Rabi":[0.48, 0.97, 0.55],
    "Bajra Kharif":[1.67, 1.67, 1.62],
    "Sugarcane":[113.38, 110.38, 107.0]
}
yield_years=[2020, 2021, 2022]
yields_list=[]
for crop, vals in yields_data.items():
    for i, y in enumerate(yield_years):
        yields_list.append({"crop_year":y, "crop":crop, "yield":vals[i]})
yields_df = pd.DataFrame(yields_list)

pune=Point(18.5204, 73.8567)
start=datetime(2020, 1, 1)
end=datetime(2022, 12, 31)
monthly=Monthly(pune, start, end).fetch().reset_index()

numeric_cols=["tavg", "tmin", "tmax", "prcp", "pres", "tsun"]
for c in numeric_cols:
    if c not in monthly.columns:
        monthly[c]=np.nan
    monthly[c]=pd.to_numeric(monthly[c], errors="coerce")

monthly["month"]=monthly["time"].dt.month
monthly["year"]=monthly["time"].dt.year

for c in numeric_cols:
    monthly[c]=monthly[c].fillna(monthly.groupby("month")[c].transform("mean"))
    monthly[c]=monthly[c].fillna(monthly[c].mean())

print("\n 36 months of monthly weather data")
print(monthly)

seasonal_rows = []
for crop, months in crop_season_mapping.items():
    for cy in yield_years:
        mask_current=(monthly["month"].isin(months)) &(monthly["year"]==cy)
        mask_next=(monthly["month"].isin([m for m in months if m <7])) & (monthly["year"]==cy+1)
        season_months = monthly[mask_current | mask_next]
        if season_months.empty:
            continue
        agg = {"tavg":season_months["tavg"].mean(), "tmin":season_months["tmin"].mean(),
            "tmax":season_months["tmax"].mean(),
            "rain":season_months["prcp"].sum(),
            "pres":season_months["pres"].mean(),
            "sunshine":season_months["tsun"].sum()
        }
        seasonal_rows.append({"crop_year":cy, "crop":crop, **agg})
seasonal_df = pd.DataFrame(seasonal_rows)

final_df = pd.merge(seasonal_df, yields_df, on=["crop_year", "crop"], how="inner")

print("\nSeasonal weather + yield dataframe")
print(final_df)

features = ["tmax", "tmin", "tavg", "rain", "sunshine"]
models = {}
for crop in final_df["crop"].unique():
    crop_df=final_df[final_df["crop"]==crop].dropna(subset=features+["yield"])
    if crop_df.shape[0]<2:
        continue
    X=crop_df[features].values
    y=crop_df["yield"].values
    model=LinearRegression()
    model.fit(X, y)
    models[crop]=model

latitude, longitude = 18.5204, 73.8567
today=datetime.now().date()
end_date=today+timedelta(days=7)
url=f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,sunrise,sunset&timezone=Asia/Kolkata&start_date={today}&end_date={end_date}"
resp=requests.get(url, timeout=15)
data=resp.json()
daily = data.get("daily", {})
fc_df = pd.DataFrame({"tmax":daily.get("temperature_2m_max", []),
    "tmin":daily.get("temperature_2m_min", []),
    "tavg":daily.get("temperature_2m_mean", []),
    "rain":daily.get("precipitation_sum", []),
    "sunshine":[max((pd.to_datetime(s)-pd.to_datetime(r)).seconds/3600, 0)
                 for r, s in zip(daily.get("sunrise", []), daily.get("sunset", []))]
})

forecast_features = {"tmax": fc_df["tmax"].mean(),
    "tmin": fc_df["tmin"].mean(),
    "tavg": fc_df["tavg"].mean(),
    "rain": fc_df["rain"].sum(),
    "sunshine": fc_df["sunshine"].sum()
}

predictions = []
for crop in yields_data.keys():
    model = models.get(crop)
    if model is None:
        pred = float("nan")
    else:
        X_new = np.array([[forecast_features[f] for f in features]])
        pred = float(model.predict(X_new)[0])
    opt = optimum_ranges.get(crop)
    suitability = {}
    if opt:
        def in_range(val, rng):
            if val is None or isnan(val):
                return False
            low, high = rng
            return low <= val <= high

        suitability["tmax_ok"] = in_range(forecast_features["tmax"], opt["tmax"])
        suitability["tmin_ok"] = in_range(forecast_features["tmin"], opt["tmin"])
        suitability["rain_ok"] = in_range(forecast_features["rain"] * 30, opt["rain"])
        suitability["sun_ok"] = in_range(forecast_features["sunshine"] * 30, opt["sunshine"])
    else:
        suitability = None

    predictions.append({"crop": crop, "predicted_yield": pred, "suitability": suitability})
res_df=pd.DataFrame(predictions)
res_df= res_df.sort_values(by="predicted_yield", ascending=False, na_position="last").reset_index(drop=True)

print("\nCrop ranking (predicted yield):")
for i, r in res_df.iterrows():
    crop=r["crop"]
    pred=r["predicted_yield"]
    suit=r["suitability"]
    suit_str =""
    if suit is None:
        suit_str=" (no optimum ranges)"
    else:
        suit_str="[" + ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in suit.items()) + "]"
    print(f"{i + 1:2d}. {crop:20s} Predicted yield = {pred:.3f}{suit_str}")

print("\nDone.") #Weather alerts for 1 to 3days
short_url=f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max&timezone=Asia/Kolkata&start_date={today}&end_date={today + timedelta(days=3)}"
short_resp=requests.get(short_url, timeout=15)
short_data= short_resp.json()
short_fc=pd.DataFrame(short_data.get("daily", {}))

def weather_alerts(short_fc):
    alerts=[]
    for i, row in short_fc.iterrows():
        date=row.get("time", f"Day {i+1}")
        tmax, tmin, rain, wind = row["temperature_2m_max"], row["temperature_2m_min"], row["precipitation_sum"], row["windspeed_10m_max"]

        day_alerts=[]
        if rain>30:
            day_alerts.append("CAUTION: Heavy rain – Delay sowing & avoid pesticide spraying.")
        if tmin<12:
            day_alerts.append("CAUTION: Cold – Consider protective irrigation or delay germination-sensitive crops.")
        if tmax>38:
            day_alerts.append("CAUTION: Very hot – Avoid sowing & irrigate if crop already planted.")
        if wind>40:
            day_alerts.append("CAUTION: Strong winds – Avoid pesticide spraying, it may drift away.")
        if not day_alerts:
            day_alerts.append("Weather normal – Safe for farming operations.")
        alerts.append({"date": date, "alerts": day_alerts})
    return alerts

alerts = weather_alerts(short_fc)
print("\nWeather Alerts & Farming Advice (Next 3 Days)")
for a in alerts:
    print(f"\nDate: {a['date']}")
    for msg in a["alerts"]:
        print(" -", msg)
