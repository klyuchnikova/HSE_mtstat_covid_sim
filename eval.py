import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import os
from datetime import datetime


def calculate_metrics(observed, predicted_mean, predicted_hdi):
    """
    Вычисляет метрики качества прогноза.
    
    Parameters:
    -----------
    observed : array-like
        Наблюдаемые значения
    predicted_mean : array-like  
        Средние предсказанные значения
    predicted_hdi : array-like
        HDI интервалы (lower, higher)
    
    Returns:
    --------
    dict : Словарь с метриками
    """
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(observed - predicted_mean))
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((observed - predicted_mean) ** 2))
    
    # Coverage (доля наблюдений внутри HDI)
    if predicted_hdi is not None:
        lower, higher = predicted_hdi
        coverage = np.mean((observed >= lower) & (observed <= higher))
    else:
        coverage = None
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((observed - predicted_mean) / (observed + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse, 
        'Coverage': coverage,
        'MAPE': mape
    }


def plot_forecast_comparison(observed, predicted_mean, predicted_hdi, dates, title="Forecast Comparison", 
                           country="Unknown", save_path=None):
    """
    Строит график сравнения наблюдений и прогнозов.
    """
    plt.figure(figsize=(12, 6))
    
    # Наблюдаемые данные
    plt.plot(dates, observed, 'ko-', label='Observed', markersize=3)
    
    # Предсказанные данные
    plt.plot(dates, predicted_mean, 'b-', label='Predicted mean', linewidth=2)
    
    # HDI интервалы
    if predicted_hdi is not None:
        lower, higher = predicted_hdi
        plt.fill_between(dates, lower, higher, alpha=0.3, color='blue', label='95% HDI')
    
    plt.title(title)
    plt.ylabel('New Cases')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Сохраняем график
    if save_path is None:
        save_path = f"results/{country}_forecast_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_path}")
    plt.show()


def plot_rt_timeline(rt_samples, dates, title="R(t) Timeline", country="Unknown", save_path=None):
    """
    Строит график временной динамики R(t).
    """
    rt_mean = rt_samples.mean(dim=['chain', 'draw'])
    rt_hdi = az.hdi(rt_samples, hdi_prob=0.95)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, rt_mean, 'b-', label='R(t) mean', linewidth=2)
    plt.fill_between(dates, rt_hdi['r_t'].sel(hdi='lower'), 
                     rt_hdi['r_t'].sel(hdi='higher'), alpha=0.3, color='blue', label='95% HDI')
    plt.axhline(y=1.0, color='red', linestyle='--', label='R=1')
    plt.title(title)
    plt.ylabel('R(t)')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Сохраняем график
    if save_path is None:
        save_path = f"results/{country}_rt_timeline.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_path}")
    plt.show()


def print_model_summary(trace):
    """
    Выводит краткую сводку по модели.
    """
    print("=== МОДЕЛЬНАЯ ДИАГНОСТИКА ===")
    print(az.summary(trace))
    
    print("\n=== КОНВЕРГЕНЦИЯ ===")
    rhat = az.rhat(trace)
    print(f"Rhat (должен быть < 1.01): {rhat.max().values:.4f}")
    
    ess = az.ess(trace)
    print(f"ESS (должен быть > 400): {ess.min().values:.0f}")
    
    # Проверяем divergences
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        n_divergences = trace.sample_stats['diverging'].sum().values
        print(f"Divergences: {n_divergences}")
    else:
        print("Divergences: информация недоступна")


def save_model_summary(trace, country="Unknown", save_path=None):
    """
    Сохраняет детальную сводку по модели в файл.
    """
    if save_path is None:
        save_path = f"results/{country}_model_summary.txt"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"=== ОТЧЕТ ПО МОДЕЛИ COVID-19: {country.upper()} ===\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== ПАРАМЕТРЫ МОДЕЛИ ===\n")
        summary_df = az.summary(trace)
        f.write(str(summary_df))
        f.write("\n\n")
        
        f.write("=== ДИАГНОСТИКА КОНВЕРГЕНЦИИ ===\n")
        rhat = az.rhat(trace)
        f.write(f"Rhat (должен быть < 1.01): {rhat.max().values:.4f}\n")
        
        ess = az.ess(trace)
        f.write(f"ESS (должен быть > 400): {ess.min().values:.0f}\n")
        
        if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
            n_divergences = trace.sample_stats['diverging'].sum().values
            f.write(f"Divergences: {n_divergences}\n")
        else:
            f.write("Divergences: информация недоступна\n")
        
        f.write("\n=== ИНТЕРПРЕТАЦИЯ ===\n")
        f.write("- Rhat близко к 1.0: цепочки хорошо смешались\n")
        f.write("- ESS > 400: достаточно эффективных образцов\n")
        f.write("- Divergences = 0: нет проблем с sampling'ом\n")
    
    print(f"Отчет сохранен: {save_path}")


def save_metrics_report(metrics, country="Unknown", save_path=None):
    """
    Сохраняет отчет с метриками в файл.
    """
    if save_path is None:
        save_path = f"results/{country}_metrics_report.txt"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"=== ОТЧЕТ ПО МЕТРИКАМ: {country.upper()} ===\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== МЕТРИКИ КАЧЕСТВА ПРОГНОЗА ===\n")
        for metric, value in metrics.items():
            if value is not None:
                if metric == 'Coverage':
                    f.write(f"{metric}: {value:.1%}\n")
                elif metric == 'MAPE':
                    f.write(f"{metric}: {value:.2f}%\n")
                else:
                    f.write(f"{metric}: {value:.2f}\n")
            else:
                f.write(f"{metric}: N/A\n")
        
        f.write("\n=== ИНТЕРПРЕТАЦИЯ МЕТРИК ===\n")
        f.write("- MAE (Mean Absolute Error): средняя абсолютная ошибка\n")
        f.write("- RMSE (Root Mean Square Error): корень из средней квадратичной ошибки\n")
        f.write("- Coverage: доля наблюдений внутри 95% доверительного интервала\n")
        f.write("- MAPE (Mean Absolute Percentage Error): средняя абсолютная процентная ошибка\n")
    
    print(f"Отчет с метриками сохранен: {save_path}")


def save_trace_data(trace, country="Unknown", save_path=None):
    """
    Сохраняет trace данные в различных форматах.
    """
    if save_path is None:
        base_path = f"results/{country}_trace"
    else:
        base_path = save_path.replace('.nc', '')
    
    # Сохраняем в NetCDF (ArviZ формат)
    nc_path = f"{base_path}.nc"
    az.to_netcdf(trace, nc_path)
    print(f"Trace данные сохранены: {nc_path}")
    
    # Сохраняем основные параметры в CSV
    csv_path = f"{base_path}_summary.csv"
    summary_df = az.summary(trace)
    summary_df.to_csv(csv_path)
    print(f"Сводка параметров сохранена: {csv_path}")
    
    return nc_path, csv_path
