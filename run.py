import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from data import load_covid_data, get_available_countries
from model import build_model, fit_model, forecast, get_rt_estimates
from eval import (calculate_metrics, plot_forecast_comparison, plot_rt_timeline, 
                 print_model_summary, save_model_summary, save_metrics_report, save_trace_data)

# === 1. Загружаем данные ===
country = "Russia"
print(f"Доступные страны: {get_available_countries()[:10]}...")  # Показываем первые 10 стран

data = load_covid_data(country, buffer_days=10)
print(f"Данные загружены: {len(data)} дней для {country}")
print(f"Период: {data.index.min().date()} - {data.index.max().date()}")
print(f"Максимум новых случаев: {data['new_cases'].max():.0f}")

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
plot_rt_timeline(rt_samples, data.index, f"R(t) Timeline — {country}", country=country)

# === 7. Posterior Predictive Check ===
with model:
    ppc = pm.sample_posterior_predictive(trace)

# Строим график сравнения наблюдений и предсказаний
observed = data["new_cases"].values
predicted_mean = ppc.posterior_predictive["observed"].mean(dim=['chain', 'draw']).values
predicted_hdi = az.hdi(ppc.posterior_predictive["observed"], hdi_prob=0.95)

plot_forecast_comparison(
    observed, predicted_mean, 
    (predicted_hdi["observed"].sel(hdi='lower').values, 
     predicted_hdi["observed"].sel(hdi='higher').values),
    data.index, 
    f"Posterior Predictive Check — {country}",
    country=country
)

# === 8. Вычисляем метрики ===
metrics = calculate_metrics(observed, predicted_mean, 
                          (predicted_hdi["observed"].sel(hdi='lower').values, 
                           predicted_hdi["observed"].sel(hdi='higher').values))
print(f"\n=== МЕТРИКИ КАЧЕСТВА ===")
for metric, value in metrics.items():
    if value is not None:
        if metric == 'Coverage':
            print(f"{metric}: {value:.1%}")
        elif metric == 'MAPE':
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.2f}")

# === 9. Модельная диагностика ===
print_model_summary(trace)

# === 10. Сохранение всех результатов ===
print(f"\n=== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===")

# Сохраняем trace данные
save_trace_data(trace, country=country)

# Сохраняем отчеты
save_model_summary(trace, country=country)
save_metrics_report(metrics, country=country)

print(f"Все результаты сохранены в папке 'results/' для страны {country}")
