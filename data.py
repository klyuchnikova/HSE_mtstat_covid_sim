import pandas as pd
import numpy as np

def load_covid_data(country: str, path: str = "covid_data.csv", buffer_days: int = 10):
    """
    Загружает данные для указанной страны
    - Сортируем по дате 
    - Оставляем только дни, где >= 100 случаев.
    - Добавляем buffer_days перед первым днём (инфекции могли начаться раньше).
    - Гарантируем отсутствие отрицательных значений.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["country"] == country].copy()
    df = df.sort_values("date").reset_index(drop=True)
    start_index = df.loc[df["new_cases"] >= 100].index.min()
    df = df.loc[start_index:].reset_index(drop=True)

    first_date = df["date"].iloc[0]
    buffer_dates = pd.date_range(start=first_date - pd.Timedelta(days=buffer_days),
                                 end=first_date - pd.Timedelta(days=1))
    buffer_df = pd.DataFrame({
        "date": buffer_dates,
        "new_cases": [0] * len(buffer_dates),
        "total_tests": [0] * len(buffer_dates),
        "country": country
    })

    df = pd.concat([buffer_df, df], ignore_index=True)
    df = df.set_index("date")
    df["new_cases"] = df["new_cases"].clip(lower=0)

    return df
