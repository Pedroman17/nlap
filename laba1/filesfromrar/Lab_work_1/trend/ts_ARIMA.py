'''
Приклад апроксимації часового ряду з алгоритмом ARIMA - потребує оптимізації параметрів
'''


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')


def main_ARIMA(data, filename):


    # 1. Підготуйте дані вашого часового ряду
    index = np.zeros((len(data)))
    for i in range(len(data)):
        index[i] = i

    # Збирання до купки часового ряду
    time_series = pd.Series(data, index=index)

    # Розділення даних на навчальні та тестові набори (необов'язково, але гарна практика)
    train_size = int(len(time_series) * 0.7)
    train_data, test_data = time_series[:train_size], time_series[train_size:]


    # 2. Визначення порядку ARIMA (p, d, q). Це важливий крок, який визначає ефективність ARIMA
    #  В прикладі задано константою, але можна автоматизувати:
    parameter = pm.auto_arima(train_data, seasonal=True, m=30,
                          start_p=0, start_q=0,
                          max_p=10, max_q=10,
                          start_P=0, start_Q=0,
                          max_D=10, max_Q=10,
                          trace=True,  # Shows the progress of the search
                          error_action='ignore',  # Ignores errors for some model fits
                          suppress_warnings=True,  # Suppresses convergence warnings
                          stepwise=True)  # Uses a stepwise search to find the best model

    order = (1, 2, 1)
    order = (0, 1, 0)

    # 3.Формування моделі ARIMA
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()

    # КРІ моделі
    print(model_fit.summary())

    # 4.Прогнозування

    forecast_steps = len(test_data)
    forecast = model_fit.forecast(steps=forecast_steps)

    # 5. Візуалізатор
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
    plt.plot(forecast.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
    plt.title('ARIMA Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


    return  forecast

