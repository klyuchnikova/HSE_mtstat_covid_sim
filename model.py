import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from scipy import stats as sps
import pytensor
import pytensor.tensor as pt
import jax.numpy as jnp
from jax.scipy.signal import convolve

def jax_conv2d(inputs, filters):
    """2D convolution using JAX that works with PyTensor"""
    return convolve(inputs, filters, mode='full', method='auto')

def get_generation_time_interval():
    """
    Возвращает дискретное распределение времени генерации инфекции (generation interval).
    Источник параметров: Systrom et al.
    """
    mean_si = 4.7
    std_si = 2.9

    mu_si = np.log(mean_si ** 2 / np.sqrt(std_si ** 2 + mean_si ** 2))
    sigma_si = np.sqrt(np.log(std_si ** 2 / mean_si ** 2 + 1))

    dist = sps.lognorm(scale=np.exp(mu_si), s=sigma_si)
    g_range = np.arange(0, 20)
    gt = np.diff(dist.cdf(g_range), prepend=0)
    gt /= gt.sum()
    return gt


def get_delay_distribution():
    """
    Возвращает распределение задержки от заражения до подтверждения.
    Используем гамма-распределение как в оригинальной модели.
    Константы взяты из начала ковида
    """
    alpha = 5.48
    beta = 0.77
    delay_range = np.arange(20)
    p_delay = sps.gamma.pdf(delay_range, alpha, scale=1.0/beta)
    p_delay /= p_delay.sum()  # Нормализуем
    return p_delay


def build_model(observed_df):
    """
    Создаёт PyMC-модель, аналогичную Twiecki COVID model, но в PyMC v5.
    """
    observed_positive = observed_df["new_cases"].values
    len_obs = len(observed_positive)
    gt = get_generation_time_interval()
    p_delay = get_delay_distribution()
    
    # Подготавливаем матрицу для свертки (как в оригинальной модели)
    convolution_ready_gt = np.zeros((len_obs - 1, len_obs))
    for t in range(1, len_obs):
        begin = max(0, t - len(gt) + 1)
        slice_update = gt[1 : t - begin + 1][::-1]
        convolution_ready_gt[t - 1, begin : begin + len(slice_update)] = slice_update
    
    convolution_ready_gt = pt.as_tensor_variable(convolution_ready_gt)

    coords = {"date": observed_df.index.values}

    with pm.Model(coords=coords) as model:

        # ----- 1. Модель динамики Rt -----
        # Rt моделируется как случайное блуждание в лог-пространстве
        log_r_t = pm.GaussianRandomWalk("log_r_t", sigma=0.035, dims="date")
        r_t = pm.Deterministic("r_t", pm.math.exp(log_r_t), dims="date")

        # ----- 2. Начальные инфекции -----
        seed = pm.Exponential("seed", 1 / 0.02)
        
        # ----- 3. Генеративный процесс (свертка) -----
        # Используем pytensor.scan для рекурсивного вычисления инфекций
        def infection_step(t, gt_matrix, y, r_t):
            return pt.sum(r_t * y * gt_matrix[t-1])
        
        y0 = pt.zeros(len_obs)
        y0 = pt.set_subtensor(y0[0], seed)
        
        outputs, _ = pytensor.scan(
            fn=infection_step,
            sequences=[pt.arange(1, len_obs)],
            outputs_info=y0,
            non_sequences=[convolution_ready_gt, r_t],
            n_steps=len_obs - 1,
        )
        infections = pm.Deterministic("infections", outputs[-1], dims="date")

        # ----- 4. Задержка до подтверждения -----
        # Свертка инфекций с распределением задержки
        test_adjusted_positive = pm.Deterministic(
            "test_adjusted_positive",
            jax_conv2d(
                infections.reshape((1, len_obs)),
                p_delay.reshape((1, len(p_delay)))
            )[0, :len_obs],
            dims="date"
        )

        # ----- 5. Exposure adjustment (если есть данные о тестах) -----
        if "total_tests" in observed_df.columns:
            tests = pm.Data("tests", observed_df["total_tests"].values, dims="date")
            exposure = pm.Deterministic(
                "exposure",
                pm.math.clip(tests, observed_df["total_tests"].max() * 0.1, 1e9),
                dims="date"
            )
            positive = pm.Deterministic("positive", exposure * test_adjusted_positive, dims="date")
        else:
            positive = test_adjusted_positive

        # ----- 6. Наблюдательная модель -----
        alpha = pm.Gamma("alpha", mu=6, sigma=1)
        pm.NegativeBinomial(
            "observed",
            mu=positive,
            alpha=alpha,
            observed=observed_positive,
            dims="date"
        )

    return model


def fit_model(model, draws=1000, tune=1000):
    with model:
        trace = pm.sample(draws=draws, tune=tune, target_accept=0.95, chains=4, cores=4)
    return trace


def forecast(model, trace, forecast_days=14):
    """Делает posterior predictive check и прогноз"""
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    return ppc


def get_rt_estimates(trace):
    """Извлекает оценки R(t) из trace"""
    return trace.posterior["r_t"]


def get_infections_estimates(trace):
    """Извлекает оценки инфекций из trace"""
    return trace.posterior["infections"]
