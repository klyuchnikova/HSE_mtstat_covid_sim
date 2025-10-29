import pandas as pd
import numpy as np
import os

def load_covid_data(country: str, path: str = None, buffer_days: int = 10):
    """
    Загружает данные для указанной страны из Our World in Data.
    - Сортируем по дате 
    - Оставляем только дни, где >= 100 случаев.
    - Добавляем buffer_days перед первым днём (инфекции могли начаться раньше).
    - Гарантируем отсутствие отрицательных значений.
    """
    # Если путь не указан, используем данные из текущей папки
    if path is None:
        path = "data/owid-covid-data.csv"
    
    # Проверяем существование файла
    if not os.path.exists(path):
        print(f"Локальный файл не найден: {path}")
        print("Загружаем данные из интернета...")
        # Загружаем данные напрямую из интернета
        url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        df = pd.read_csv(url, parse_dates=["date"])
        print("✓ Данные загружены из интернета")
    else:
        # Загружаем данные из локального файла
        df = pd.read_csv(path, parse_dates=["date"])
    
    # Фильтруем по стране
    df = df[df["location"] == country].copy()
    
    if len(df) == 0:
        raise ValueError(f"Данные для страны '{country}' не найдены")
    
    # Сортируем по дате
    df = df.sort_values("date").reset_index(drop=True)
    
    # Оставляем только дни с >= 100 случаев
    start_index = df.loc[df["new_cases"] >= 100].index.min()
    if pd.isna(start_index):
        raise ValueError(f"Нет дней с >= 100 случаев для страны '{country}'")
    
    df = df.loc[start_index:].reset_index(drop=True)
    
    # Добавляем buffer_days
    first_date = df["date"].iloc[0]
    buffer_dates = pd.date_range(start=first_date - pd.Timedelta(days=buffer_days),
                                 end=first_date - pd.Timedelta(days=1))
    buffer_df = pd.DataFrame({
        "date": buffer_dates,
        "new_cases": [0] * len(buffer_dates),
        "total_tests": [0] * len(buffer_dates),
        "location": country
    })

    df = pd.concat([buffer_df, df], ignore_index=True)
    df = df.set_index("date")
    
    # Очищаем данные
    df["new_cases"] = df["new_cases"].fillna(0).clip(lower=0)
    df["total_tests"] = df["total_tests"].fillna(0).clip(lower=0)
    
    # Переименовываем колонки для совместимости
    df = df.rename(columns={"location": "country"})

    return df


def get_available_countries(path: str = None):
    """
    Возвращает список доступных стран в данных.
    """
    if path is None:
        path = "data/owid-covid-data.csv"
    
    if not os.path.exists(path):
        print(f"Локальный файл не найден: {path}")
        print("Загружаем данные из интернета...")
        # Загружаем данные напрямую из интернета
        url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        df = pd.read_csv(url)
        print("✓ Данные загружены из интернета")
    else:
        # Загружаем данные из локального файла
        df = pd.read_csv(path)
    return sorted(df["location"].unique().tolist())
