# HSE_mtstat_covid_sim

Задание:

1. Реализовать стохастическую генеративную модель как приложение на языке Python c
помощь библиотеки pymc. Не использовать предыдущую версию библиотеки -
pymc3!
2. Разъяснить в комментариях, зачем используется та или иная строчка кода
3. Обучить модель на данных о числе новых выявленных случаев заболевания covid-19
в период с 01.01.2020 по 01.12.2020 в странах: Россия, Италия, Германия и Франция.
Для каждой страны не учитывать дни до того как заболеваемость превысит 100
случаев в день!
4. Оценить динамику эффективного репродуктивного числа R(t)
5. Предсказать число зарегистрированных случаев в день и R(t) для диапазона дат
02.12.2020 по 14.12.2020. Сравнить с реальными данными.

---

# Короткая сводка задачи (цель)

* Построить стохастическую генеративную модель эпидемии (на PyMC, не pymc3) для числа новых подтверждённых случаев COVID-19 в 2020 (Россия, Италия, Германия, Франция).
* Для каждой страны отбросить дни до достижения порога 100 случаев/день.
* Оценить динамику эффективного репродуктивного числа (R(t)).
* Сделать прогноз числа случаев и (R(t)) на 2020-12-02 — 2020-12-14 и сравнить с реальными данными.

Опорные материалы: PyMC «Bayesian workflow» (работа с эпидмоделями, prior checks, PPC и пр.), пример Twiecki (generative covid model) с github. ([pymc.io][1])

---

# Общий план (высокоуровневые этапы)

1. Сбор данных + предобработка (фильтрация по порогу 100).
2. Изучение и упрощение модели Twiecki → выбор блоков для реализации (generation interval, задержка до подтверждения, модель для (R_t), сэмплинг заражений, наблюдательная модель с NegativeBinomial). ([GitHub][2])
3. Реализация модели в PyMC (v5): декларация координат, приоритеты (priors), стохастические процессы (GaussianRandomWalk или аналог) и детерминанты (infections, conv с delay).
4. Проверки до подгонки: prior predictive checks (генерация данных из prior, визуализация) — часть Bayesian workflow. ([pymc.io][1])
5. Fitting (sample) с ArviZ diagnostics (trace, Rhat, ESS, divergences).
6. Posterior predictive: прогноз на 2020-12-02..2020-12-14 и извлечение (R(t)) постериорно.
7. Сравнение прогноза с реальными данными (графики, MAE/RMSE, покрытие 95% CI).
8. Документация: комментарии к строкам кода, пояснения гиперпараметров и нюансов.

---

# Детализированный план с пояснениями «что именно и зачем»

## 1) Данные (what & why)

* Источник: WHO / JHU / предоставленный набор. Нужно получить ежедневные новые подтверждённые случаи и (если доступно) число тестов (или хотя бы total tests), индекс по дате. (Прим. Twiecki использует WHO-like и «days_since_100» фильтр.) ([pymc.io][1])
* Для каждой страны: оставить только даты начиная с первого дня, когда `new_cases >= 100`. (Это прямо в гайде — фильтрация по 100 случаев, чтобы убрать очень шумный ранний период.) ([pymc.io][1])
* Добавить buffer_days (например 10—14 дней) — модель требует начального seed до наблюдений (инфекции происходят раньше подтверждений). (Как в реализации Twiecki.) ([GitHub][2])

**Выход:** pandas/Polars DataFrame с индексом datetime и столбцами `positive` (daily new confirmed), `total` (tests) если есть, `days_since_100`.

---

## 2) Предварительная визуализация и простые статистики

* Построй график `daily new cases` и `cumulative`. Оценить сезонность/шумы/артефакты. (Этап «plot the data» гайда workflow.) ([pymc.io][1])
* Посчитать среднее/медию/дисперсию по периоду обучения и проверить выбросы.

Зачем: увидеть аномалии, понять адекватность предположений (например, экспоненциальное распространение на ранней фазе).

---

## 3) Компоненты генеративной модели (математически и на словах)

### 3.1. Generation interval (g(\tau))

* Берём дискретное распределение времени генерации (generation time, serial interval) — в Twiecki: логнорм с mean≈4.7, sd≈2.9, затем дискретизация по дням и нормировка на сумму 1. Это даёт вектор (g_1, g_2, \dots, g_L). ([GitHub][2])

### 3.2. Динамика (R_t)

* Моделируем лог-(R_t) как случайное блуждание (Gaussian Random Walk):
  [
  \log R_t \sim \text{GaussianRandomWalk}(\sigma)
  ]
  и (R_t = \exp(\log R_t)). Twiecki использует (\sigma \approx 0.035) как «насколько быстро может меняться (R_t)». ([GitHub][2])

### 3.3. Инициация (seed infections)

* Вводим `seed ~ Exponential(1/0.02)` или похожий prior — начальное число инфицированных до наблюдений. Затем строим вектор инфекций ({y_t}) рекурсивно:
  [
  y_0 = \text{seed},\qquad y_t = \sum_{\tau=1}^{L} R_t, y_{t-\tau}, g_\tau
  ]
  (Это — классическая дискретная рекуррентная формула, реализуемая свёрткой; Twiecki использует theano.scan и precomputed convolution matrix.) ([GitHub][2])

### 3.4. Задержка до подтверждения (delay distribution) и перевод инфекций → подтверждения

* Есть распределение задержки от заражения до подтверждения `p_delay` (узнаётся в `patients.get_delay_distribution()` в репозитории). Надо сделать свёртку `infections * p_delay` → ожидаемые подтверждённые случаи (на день). Twiecki использует conv2d / conv with p_delay. ([GitHub][2])

### 3.5. Adjustment by test volume / exposure

* Если есть данные по тестам, модель масштабирует ожидаемые подтверждённые случаи на показатель `exposure` (clip тестов снизу 0.1*max_tests). Это даёт `positive = exposure * test_adjusted_positive`. Это примерно как offset/exposure в Poisson-регрессии: больше тестов → выше вероятность обнаружения. ([GitHub][2])

### 3.6. Наблюдательная модель (likelihood)

* Наблюдаемые ненулевые подтверждённые случаи моделируются через Negative Binomial:
  [
  \text{observed}_t \sim \text{NegBinom}(\mu = positive_t, \alpha)
  ]
  где `alpha` — overdispersion (напр., Gamma prior with mean~6, sd~1). ([GitHub][2])

---

## 4) Как это реализовать в PyMC (структура кода и что комментировать)

1. Импорт и настройки:

   ```py
   import pymc as pm            # pymc v5
   import arviz as az
   import numpy as np
   import pandas as pd
   # + utilities: preliz, polars, plotly, etc.
   ```

   Комментарий: использовать pymc (не pymc3); ArviZ для диагностики/ppc. ([pymc.io][1])

2. Подготовка данных:

   * загрузить, отфильтровать по `>=100`, добавить buffer_days,
   * сделать `coords = {"date": dates, "nonzero_date": dates[where tests>0]}` (координаты помогают ArviZ).
     Комментарий: объяснить почему buffer_days (инфекции предшествуют подтверждениям).

3. В модели `with pm.Model(coords=coords) as model:`

   * `log_r_t = pm.GaussianRandomWalk("log_r_t", sigma=0.035, dims=["date"])`
     комментарий: случайная прогулка для лог R_t; sigma задаёт гладкость.
   * `r_t = pm.Deterministic("r_t", pm.math.exp(log_r_t), dims=["date"])`
     комментарий: экспонента для положительности R_t.
   * `seed = pm.Exponential("seed", 1/0.02)`
     комментарий: prior для стартового числа инфекций.
   * Реализация рекуррентной формулы для infections: (в PyMC/PyTensor используем `pm.scan`/`aesara.scan` или векторизованную conv с precomputed matrix как в Twiecki).
     комментарий: здесь важно объяснить, что мы симулируем «имплицитную» серийную передачу через свёртку с generation time.
   * `test_adjusted_positive = pm.Deterministic("test_adjusted_positive", conv2d(infections, p_delay)[:len_obs])`
     комментарий: конволюция infections и delay → ожидаемые подтверждения.
   * `exposure = pm.Deterministic("exposure", pm.math.clip(tests, tests.max()*0.1, 1e9), dims=["date"])`
     комментарий: защита от нулевого объёма тестирования.
   * `positive = pm.Deterministic("positive", exposure * test_adjusted_positive, dims=["date"])`
   * `alpha = pm.Gamma("alpha", mu=6, sigma=1)` — prior для overdispersion.
   * `nonzero_positive = pm.NegativeBinomial("nonzero_positive", mu=positive[nonzero_days], alpha=alpha, observed=observed_positive[nonzero_days])`
     комментарий: likelihood наблюдаемых ненулевых дней.

4. Prior predictive checks:

   * `ppc_prior = pm.sample_prior_predictive(samples=200)`
     комментарий: генерируем данные из prior, смотрим разумность (не хотим, чтобы prior генерил нереальные числа).

5. Fitting:

   ```py
   trace = pm.sample(draws=1000, tune=2000, chains=4, cores=4, target_accept=0.95)
   ```

   Комментарий: tune (адаптация) = 2000, target_accept=0.95 чтобы уменьшить divergences; настроить под задачу. Diagnostics (az.summary, az.plot_trace). ([pymc.io][1])

6. Posterior predictive (и прогноз):

   * Для прогнозов на 2020-12-02..14 нужно расширить временной ряд: в модели добавить дополнительные дни (exposed tests может быть NaN/0) либо построить генеративный шаг с фиксацией `R_t` будущих значений — обычно продолжаем R_t равным последнему оцененному или моделируем random walk продолжение и делаем posterior predictive на будущие даты (PyMC позволяет `pm.sample_posterior_predictive`). Комментарий: лучше моделировать будущее R_t как продолжение GaussianRandomWalk (так модель сама выдаст распределение R(t) в будущем).
   * `ppc = pm.sample_posterior_predictive(trace, var_names=["positive", "r_t"])` — получить предсказания для `positive` и `r_t`.

7. Оценки и метрики:

   * Для каждого дня в прогнозном окне: извлечь медиану и 95% интервал из ppc для `positive`. Сравнить с реальным `observed`.
   * Метрики: MAE, RMSE, и coverage% (доля дней, где реальное значение внутри 95% CI).
   * Отдельно: графики `R(t)` со средним и 95% CI (post median и HPD).

8. Документация к коду:

   * К каждому важному блоку/переменной (priors, conv, exposure, alpha, seed, sigma для RW) — короткий комментарий «почему такое распределение/параметр». Это требование задания.

---

## 5) Практические рекомендации и типичные проблемы (и что комментировать в отчёте)

* **Почему Negative Binomial?** — overdispersion: реальные ежедневные случаи часто имеют дисперсию > mean. Объясни выбор `alpha` prior. ([GitHub][2])
* **Почему GaussianRandomWalk для log R_t?** — даёт гладную (но стохастическую) кривую R_t; sigma контролирует, насколько быстро R_t может меняться. Объясни влияние sigma (меньше → более гладко). ([GitHub][2])
* **Prior predictive checks** — объясни, что это тестирует: «наши priors не должны генерить абсурдные данные». ([pymc.io][1])
* **Диагностика MCMC** — покажи trace plots, R̂ (Rhat) ≈1, ESS, divergences (если есть — надо менять target_accept или reparameterize). Комментируй действия, если divergences > 0 (увеличить target_accept или трансформировать модель). ([pymc.io][1])
* **Числа для sampling** — начни с `tune=2000, draws=1000, chains=4, target_accept=0.95`. Если модель медленно смешивается — увеличь `tune` и `draws`. ([pymc.io][1])

---

## 6) План файлов / модулей (что написать в коде, порядок)

1. `data.py` — загрузка и предобработка (фильтрация по 100, buffer_days), функции helper: `get_country_df(country)`. Комментарии: пояснить каждое преобразование.
2. `models.py` — класс `GenerativeModel` (в духе Twiecki), реализующий `build()` и `sample()`. Писать подробные docstrings и inline comments. ([GitHub][2])
3. `train_and_forecast.py` — прикладной скрипт: для каждой страны — строит модель, делает prior checks, fit, posterior predictive, сохраняет результаты и картинки.
4. `eval.py` — функ-ии вычисления MAE/RMSE/coverage, визуализатор R(t) и прогнозных интервалов.
5. `notebooks/` — jupyter notebook с интерактивными графиками и описанием эксперимента (полезно для демо).

---

## 7) Что комментировать по-строчно (примеры)

* `log_r_t = pm.GaussianRandomWalk("log_r_t", sigma=0.035, dims=["date"])`
  — «логарифм R_t моделируется как случайная прогулка с характерной дисперсией sigma=0.035: это определяет насколько быстро Rt может меняться; мы работаем в лог-пространстве, чтобы R_t оставался >0.»
* `seed = pm.Exponential("seed", 1/0.02)`
  — «априор на количество начальных инфицированных; экспоненциальный prior ставит массу около нуля, но допускает крупные значения.»
* `infections = ... scan/conv ...`
  — «рекурсивная свёртка: infections_t = sum_{tau} R_t * infections_{t-tau} * g_tau — моделируем передачу инфекции через поколения.»
* `nonzero_positive = pm.NegativeBinomial(... observed=observed_positive_nonzero)`
  — «likelihood: предполагаем overdispersed счетный процесс; используем NegativeBinomial, чтобы учесть варьирование и выбросы.»
* `pm.sample(... target_accept=0.95)`
  — «используем HMC/NUTS с более высоким target_accept, чтобы снизить divergences для сложной модели.»

---

## 8) Выходы для отчёта (что представить)

* Прикрепить prior predictive plots (перед подгонкой) и posterior predictive plots (после). ([pymc.io][1])
* Трассы параметров, R̂, ESS, количество divergences.
* Графики: observed vs posterior predictive (с median & 95% CI) для train и test (02–14 Dec).
* Таблица метрик (MAE, RMSE, coverage) по стране и по дням.
* График (R(t)) с 95% CI и краткий аналитический комментарий (например: «R(t) стабилизировалось около 0.9 в конце ноября»).
* Короткий блок «что можно улучшить» (например, иерархическая модель, учёт изменений в тестировании, сезонных эффектов).

---

## 9) Временная оценка (порядок выполнения)

(ориентировочно; полезно для планирования)

* 0.5 дня — подготовка окружения и установка зависимостей (pymc v5 и т.д.).
* 0.5–1 день — загрузка/предобработка данных и exploratory plots.
* 1–2 дня — реализация базовой версии модели (по Twiecki), prior predictive checks.
* 1 день — подгонка и устранение проблем (диагностика, tuning).
* 0.5–1 день — posterior predictive, прогноз и вычисление метрик.
* 0.5 дня — оформление комментариев/отчёта/графиков.

---

# Ссылки на ключевые материалы (опорные)

* PyMC — «The Bayesian Workflow: COVID-19 Outbreak Modeling» (приёмы prior checks, diagnostics, P.P.C.). ([pymc.io][1])
* Twiecki — репозиторий covid-model (реализация генеративной модели: GaussianRandomWalk, conv matrix, delay distribution, likelihood). ([GitHub][2])
* PyMC — пример «Forecasting with Structural Timeseries» (как делать posterior predictive / forecasting с PyMC). ([pymc.io][3])

[1]: https://www.pymc.io/projects/examples/en/latest/case_studies/bayesian_workflow.html "The Bayesian Workflow: COVID-19 Outbreak Modeling — PyMC example gallery"
[2]: https://github.com/twiecki/covid-model/tree/setuppy/covid/models "covid-model/covid/models at setuppy · twiecki/covid-model · GitHub"
[3]: https://www.pymc.io/projects/examples/en/latest/time_series/Forecasting_with_structural_timeseries.html "Forecasting with Structural AR Timeseries — PyMC example gallery"
