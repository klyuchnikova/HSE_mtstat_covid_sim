#!/usr/bin/env python3
"""
Скрипт для запуска анализа COVID-19 модели для всех стран из задания.
"""

import os
import time
from datetime import datetime

import pymc as pm
import arviz as az

from data import load_covid_data, get_available_countries
from model import build_model, fit_model, get_rt_estimates
from eval import (plot_rt_timeline, plot_forecast_comparison, calculate_metrics,
                 save_model_summary, save_metrics_report, save_trace_data)

# Страны из задания
TARGET_COUNTRIES = ["Russia", "Italy", "Germany", "France"]

def run_analysis_for_country(country):
    """
    Запускает полный анализ для одной страны.
    """
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ ДЛЯ СТРАНЫ: {country.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 1. Загружаем данные
        print(f"Загрузка данных для {country}...")
        data = load_covid_data(country, buffer_days=10)
        print(f"✓ Данные загружены: {len(data)} дней")
        print(f"  Период: {data.index.min().date()} - {data.index.max().date()}")
        print(f"  Максимум новых случаев: {data['new_cases'].max():.0f}")
        
        # 2. Строим модель
        print(f"Построение модели для {country}...")
        model = build_model(data)
        print("✓ Модель построена")
        
        # 3. Prior predictive check
        print(f"Prior predictive check для {country}...")
        with model:
            prior_pred = pm.sample_prior_predictive(samples=200)
        print("✓ Prior predictive check завершен")
        
        # 4. Обучение модели
        print(f"Обучение модели для {country}...")
        trace = fit_model(model, draws=500, tune=1000)
        print("✓ Модель обучена")
        
        # 5. Posterior predictive check
        print(f"Posterior predictive check для {country}...")
        with model:
            ppc = pm.sample_posterior_predictive(trace)
        print("✓ Posterior predictive check завершен")
        
        # 6. Визуализация R(t)
        print(f"Создание графиков R(t) для {country}...")
        rt_samples = get_rt_estimates(trace)
        plot_rt_timeline(rt_samples, data.index, f"R(t) Timeline — {country}", country=country)
        
        # 7. График сравнения
        print(f"Создание графиков сравнения для {country}...")
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
        
        # 8. Вычисляем метрики
        print(f"Вычисление метрик для {country}...")
        metrics = calculate_metrics(observed, predicted_mean, 
                                  (predicted_hdi["observed"].sel(hdi='lower').values, 
                                   predicted_hdi["observed"].sel(hdi='higher').values))
        
        print(f"\n=== МЕТРИКИ ДЛЯ {country.upper()} ===")
        for metric, value in metrics.items():
            if value is not None:
                if metric == 'Coverage':
                    print(f"{metric}: {value:.1%}")
                elif metric == 'MAPE':
                    print(f"{metric}: {value:.2f}%")
                else:
                    print(f"{metric}: {value:.2f}")
        
        # 9. Сохранение результатов
        print(f"Сохранение результатов для {country}...")
        save_trace_data(trace, country=country)
        save_model_summary(trace, country=country)
        save_metrics_report(metrics, country=country)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Анализ для {country} завершен за {elapsed_time:.1f} секунд")
        
        return {
            'country': country,
            'success': True,
            'elapsed_time': elapsed_time,
            'metrics': metrics,
            'data_points': len(data)
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Ошибка при анализе {country}: {str(e)}")
        return {
            'country': country,
            'success': False,
            'elapsed_time': elapsed_time,
            'error': str(e)
        }

def main():
    """
    Основная функция для запуска анализа всех стран.
    """
    print("COVID-19 ГЕНЕРАТИВНАЯ МОДЕЛЬ - АНАЛИЗ ВСЕХ СТРАН")
    print("=" * 60)
    print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Целевые страны: {', '.join(TARGET_COUNTRIES)}")
    
    # Проверяем доступные страны
    available_countries = get_available_countries()
    print(f"Доступно стран в данных: {len(available_countries)}")
    
    # Проверяем наличие целевых стран
    missing_countries = [c for c in TARGET_COUNTRIES if c not in available_countries]
    if missing_countries:
        print(f"⚠️  ВНИМАНИЕ: Не найдены данные для: {missing_countries}")
    
    # Создаем папку для результатов
    os.makedirs("results", exist_ok=True)
    
    # Запускаем анализ для каждой страны
    results = []
    total_start_time = time.time()
    
    for country in TARGET_COUNTRIES:
        if country in available_countries:
            result = run_analysis_for_country(country)
            results.append(result)
        else:
            print(f"⚠️  Пропускаем {country} - данные недоступны")
    
    # Итоговый отчет
    total_elapsed = time.time() - total_start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print("ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*60}")
    print(f"Время выполнения: {total_elapsed:.1f} секунд")
    print(f"Успешно обработано: {len(successful)}/{len(TARGET_COUNTRIES)} стран")
    
    if successful:
        print(f"\n✓ УСПЕШНЫЕ АНАЛИЗЫ:")
        for result in successful:
            print(f"  - {result['country']}: {result['elapsed_time']:.1f}с, {result['data_points']} точек")
    
    if failed:
        print(f"\n✗ ОШИБКИ:")
        for result in failed:
            print(f"  - {result['country']}: {result['error']}")
    
    print(f"\nВсе результаты сохранены в папке 'results/'")
    print(f"Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
