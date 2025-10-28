import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from data import load_covid_data
from model import build_model, fit_model, forecast, get_rt_estimates
from eval import (calculate_metrics, plot_forecast_comparison, plot_rt_timeline, 
                 print_model_summary, save_model_summary, save_metrics_report, save_trace_data)

# === 1. Загружаем данные ===
country = "Russia"
data = load_covid_data(country, path="covid_data.csv", buffer_days=10)
print(f"Данные загружены: {len(data)} дней для {country}")

# === 2. Строим модель ===
model = build_model(data)
print("Модель построена.")

# === 3. Проверяем приоры (prior predictive check) ===
with model:
    prior_pred = pm.sample_prior_predictive(samples=200)

az.plot_ppc(prior_pred)
plt.title("Prior predictive check")
plt.show()

# === 4. Обучаем модель ===
trace = fit_model(model, draws=500, tune=1000)
az.summary(trace)

# === 5. Posterior predictive (прогноз) ===
ppc = forecast(model, trace, forecast_days=14)

az.plot_ppc(ppc, num_pp_samples=100)
plt.title(f"Posterior predictive — {country}")
plt.show()

# === 6. Визуализация R(t) ===
rt_samples = get_rt_estimates(trace)
plot_rt_timeline(rt_samples, data.index, f"R(t) Timeline — {country}")

# === 7. Модельная диагностика ===
print_model_summary(trace)

# === 8. Сохранение результатов ===
trace_path = f"results/{country}_trace.nc"
az.to_netcdf(trace, trace_path)
print(f"Результаты сохранены в {trace_path}")
