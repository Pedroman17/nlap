'''
Модуль нейромережного оброблення часового ряду
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def file_parsing(File_name="data/Oschadbank (USD).xls", Data_name="Купівля"):

    d = pd.read_excel(File_name)
    S_real = d[[Data_name]].values.reshape(-1, )
    initial_iter = len(S_real)
    zeros_count = initial_iter - np.count_nonzero(S_real)

    if zeros_count / initial_iter < 0.2:
        S_real = S_real[np.nonzero(S_real)]
        print("Оскільки нульових значень менше 20% - їх було видалено")

    # recalculate_iter = len(SL0)
    print('Кількість нулів=', zeros_count, " у %= ", round(100 * zeros_count / initial_iter, 3), "%")

    return S_real


# Створення моделі
class MyModel(tf.keras.Model):

    def __init__(self, n_stocks, n_neurons, sigma):
        super(MyModel, self).__init__()
        weight_initializer = tf.keras.initializers.VarianceScaling(seed=7, scale=sigma, mode="fan_avg",
                                                                   distribution="uniform")
        bias_initializer = tf.zeros_initializer()
        self.hidden_layers = [tf.keras.layers.Dense(n_neurons[i], activation=tf.nn.relu,
                                                    kernel_initializer=weight_initializer,
                                                    bias_initializer=bias_initializer)
                              for i in range(len(n_neurons))]
        self.out = tf.keras.layers.Dense(1, kernel_initializer=weight_initializer,
                                         bias_initializer=bias_initializer)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out(x)


def main_nnet(batch_size, epochs, real_data, n_neurons):

    n_neurons = [1024, 512, 256, 64] if n_neurons is None else n_neurons

    m = len(real_data)
    n = 500
    S = (np.random.randn(n))

    train_data = np.zeros((m, n))
    test_data = np.zeros((m, n))
    for j in range(m):
        for i in range(n):
            train_data[j, i] = (real_data[j]) + S[i]  # модель реального процесу
            test_data[j, i] = (real_data[j]) + S[i]
    plt.plot(train_data)
    plt.title('Вхідні параметри для нейроної мережі')
    plt.show()

    # Нормалізація даних
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Масив даних x та y
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]

    # Формування параметрів нейромережі
    n_stocks = X_train.shape[1]
    sigma = 1

    # Створюємо об'єкт моделі
    model = MyModel(n_stocks, n_neurons, sigma)

    # Оптимізатор та функція втрат
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='MeanSquaredError', optimizer=optimizer)

    # Тренування моделі
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Прогнозуємо на тестовому наборі даних
    predicted = model.predict(X_test)

    # Графік результату
    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(y_test.shape[0]), y_test, color='blue', label='Real')
    plt.plot(np.arange(y_test.shape[0]), predicted, color='red', label='Predicted')
    plt.title('Real vs Predicted')
    plt.legend()
    plt.show()

    return predicted



if __name__ == '__main__':

    # Параметри мережі
    batch_size = 256
    epochs = 5
    n_neurons = [499, 312, 156, 64]
    real_data = file_parsing()
    main_nnet(batch_size, epochs, real_data, n_neurons)


