'''
Блок головних викликів
'''

import sys
import math as mt
from data_parser import file_parsing
from data_model import Model, randomAM, randoNORM, Model_NORM, Model_NORM_AV
from data_analytics import Plot_AV,  Stat_characteristics_in, Stat_characteristics_out, Stat_characteristics_extrapol, Stat_characteristics_out_expo
from ts_statistical_learning import Sliding_Window_AV_Detect_medium, Sliding_Window_AV_Detect_LSM, Sliding_Window_AV_Detect_sliding_wind
from ts_statistical_learning import LSM, ABF, LSM_Extrapol, LSM_exponent, Expo_Regres
from ts_statistical_learning import r2_score, r2_score_expo
from ts_neural_networks import main_nnet
from ts_ARIMA import main_ARIMA


if __name__ == '__main__':

    path_plot = 'data_plot/'
    path_data = 'data/'

    filename_hist = path_plot + 'data_hist_in'

    # Джерело вхідних даних
    print('Оберіть джерело вхідних даних та подальші дії:')
    print('1 - модель')
    print('2 - реальні дані')
    print('3 - Бібліотеки для статистичного навчання -->>> STOP')
    Data_mode = int(input('mode:'))

    if (Data_mode == 1):
        # АРІ констант
        n = 10_000
        iter = int(n)  # кількість реалізацій ВВ
        Q_AV = 3  # коефіцієнт переваги АВ
        nAVv = 10
        nAV = int((iter * nAVv) / 100)  # кількість АВ у відсотках та абсолютних одиницях
        dm = 0
        dsig = 5  # параметри нормального закону розподілу ВВ: середне та СКВ

        # модель даних
        S0 = Model(n)  # модель ідеального тренду (квадратичний закон)
        SAV = randomAM(n, iter, nAV)  # модель рівномірних номерів АВ
        S = randoNORM(dm, dsig, iter)  # модель нормальних помилок
        # Нормальні похибки
        SV = Model_NORM(S, S0, n)  # модель тренда + нормальних помилок
        Plot_AV(S0, SV, 'квадратична модель + Норм. шум', path_plot + 'Model_NORM.png')
        Stat_characteristics_in(SV, 'Вибірка + Норм. шум', filename_hist)
        # Аномальні похибки
        SV_AV = Model_NORM_AV(S0, SV, nAV, Q_AV, dm, dsig, SAV) # модель тренда + нормальних помилок + АВ
        # аналіз вхідних даних
        Plot_AV(S0, SV_AV, 'квадратична модель + Норм. шум + АВ', path_plot + 'Model_NORM_AV.png')
        Stat_characteristics_in(SV_AV, 'Вибірка з АВ', filename_hist)

    if (Data_mode == 2):

        # вибіріть реальні дані
        SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', path_data + 'Oschadbank (USD).xls', 'Купівля')  # реальні дані
        # SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', path_data + 'Oschadbank (USD).xls', 'Продаж')  # реальні дані
        # SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', path_data + 'Oschadbank (USD).xls', 'КурсНбу')  # реальні дані
        n = len(SV_AV)
        # аналіз вхідних даних
        iter = int(len(SV_AV))  # кількість реалізацій ВВ
        Plot_AV(SV_AV, SV_AV, 'Коливання курсу USD в 2022 році за даними Ощадбанк', path_plot + 'data_real.png')
        Stat_characteristics_in(SV_AV, 'Коливання курсу USD в 2022 році за даними Ощадбанк', filename_hist)

    if (Data_mode == 3):
        print('Бібліотеки Python для реалізації методів статистичного навчання:')
        print('https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html')
        print('https://scikit-learn.org/stable/modules/sgd.html#regression')
        sys.exit(0)


    # обробка вхідних даних

    print('Оберіть функціонал процесів навчання:')
    print('1 - детекція та очищення від АВ: метод medium')
    print('2 - детекція та очищення від АВ: метод MNK')
    print('3 - детекція та очищення від АВ: метод sliding window')
    print('4 - AB фільтрація')
    print('5 - МНК згладжування')
    print('6 - МНК прогнозування')
    print('7 - МНК експонента за R&D')
    print('8 - Експонента за класикою')
    print('9 - Нейромережа')
    print('10 - ARIMA')

    mode = int(input('mode:'))

    if (mode == 1):
        print('Вибірка очищена від АВ метод medium')
        # Увага!!! якість результату залежить від параметрів еталонного вікна та чутливості Q
        N_Wind_Av = 5  # розмір ковзного вікна для виявлення АВ
        Q = 1.6  # коефіцієнт виявлення АВ
        S_AV_Detect_medium = Sliding_Window_AV_Detect_medium(SV_AV, N_Wind_Av, Q)
        Stat_characteristics_in(S_AV_Detect_medium, 'Вибірка очищена від алгоритм medium АВ', filename_hist)
        Yout_SV_AV_Detect = LSM(S_AV_Detect_medium)
        Stat_characteristics_out(SV_AV, Yout_SV_AV_Detect, 'МНК Вибірка відчищена від АВ алгоритм medium')
        Plot_AV(S0, S_AV_Detect_medium, 'Вибірка очищена від АВ алгоритм medium', path_plot + '1_Detect_medium.png')

    if (mode == 2):
        print('Вибірка очищена від АВ метод MNK')
        # Очищення від аномальних похибок МНК
        # Увага!!! якість результату залежить від параметрів налаштувань n_Wind Q_MNK
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        Q_MNK = 7  # коефіцієнт виявлення АВ
        S_AV_Detect_MNK = Sliding_Window_AV_Detect_LSM(SV_AV, Q_MNK, n_Wind)
        Stat_characteristics_in(S_AV_Detect_MNK, 'Вибірка очищена від АВ алгоритм MNK', filename_hist)
        Yout_SV_AV_Detect_MNK = LSM(S_AV_Detect_MNK)
        Stat_characteristics_out(SV_AV, Yout_SV_AV_Detect_MNK, 'МНК Вибірка очищена від АВ алгоритм MNK')
        Plot_AV(S0, S_AV_Detect_MNK, 'Вибірка очищена від АВ алгоритм MNK', path_plot + '2_Detect_MNK.png')

    if (mode == 3):
        print('Вибірка очищена від АВ метод sliding_wind')
        # Очищення від аномальних похибок sliding window
        # Увага!!! якість результату залежить від n_Wind
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', filename_hist)
        Yout_SV_AV_Detect_sliding_wind = LSM(S_AV_Detect_sliding_wind)
        Stat_characteristics_out(SV_AV, Yout_SV_AV_Detect_sliding_wind, 'МНК Вибірка очищена від АВ алгоритм sliding_wind')
        Plot_AV(S0, S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', path_plot + '3_Detect_sliding_wind.png')

    if (mode == 4):
        print('ABF згладжена вибірка очищена від АВ алгоритм sliding_wind')
        # Очищення від аномальних похибок sliding window
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', filename_hist)
        Yout_SV_AV_Detect_sliding_wind = ABF(S_AV_Detect_sliding_wind)
        Stat_characteristics_out(SV_AV, Yout_SV_AV_Detect_sliding_wind,
                             'ABF згладжена, вибірка очищена від АВ алгоритм sliding_wind')
        # Оцінювання якості моделі та візуалізація
        r2_score(S_AV_Detect_sliding_wind, Yout_SV_AV_Detect_sliding_wind, 'ABF_модель_згладжування')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
                'ABF Вибірка очищена від АВ алгоритм sliding_wind', path_plot + '4_ABF.png')

    if (mode == 5):
        print('MNK згладжена вибірка очищена від АВ алгоритм sliding_wind')
        # Очищення від аномальних похибок sliding window
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', filename_hist)
        Yout_SV_AV_Detect_sliding_wind = LSM(S_AV_Detect_sliding_wind)
        Stat_characteristics_out(SV_AV, Yout_SV_AV_Detect_sliding_wind,
                             'MNK згладжена, вибірка очищена від АВ алгоритм sliding_wind')
        # Оцінювання якості моделі та візуалізація
        r2_score(S_AV_Detect_sliding_wind, Yout_SV_AV_Detect_sliding_wind, 'MNK_модель_згладжування')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
                'MNK Вибірка очищена від АВ алгоритм sliding_wind', path_plot + '5_LSM.png')

    if (mode == 6):
        print('MNK ПРОГНОЗУВАННЯ')
        # Очищення від аномальних похибок sliding window
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
        koef = mt.ceil(n * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки
        S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', filename_hist)
        Yout_SV_AV_Detect_sliding_wind = LSM_Extrapol(S_AV_Detect_sliding_wind, koef)
        Stat_characteristics_extrapol(koef, Yout_SV_AV_Detect_sliding_wind,
                             'MNK ПРОГНОЗУВАННЯ, вибірка очищена від АВ алгоритм sliding_wind')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
                'MNK ПРОГНОЗУВАННЯ: Вибірка очищена від АВ алгоритм sliding_wind', path_plot + '6_LSM_predicted.png')

    if (mode == 7):
        print('MNK ЕКСПОНЕНТА')
        # Очищення від аномальних похибок sliding window
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
        koef = mt.ceil(n * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки
        S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', filename_hist)
        Yout_SV_AV_Detect_sliding_wind = LSM_exponent(S_AV_Detect_sliding_wind)
        Stat_characteristics_out(SV_AV, Yout_SV_AV_Detect_sliding_wind,
                             'MNK ЕКСПОНЕНТА, вибірка очищена від АВ алгоритм sliding_wind')
        #  Оцінювання якості моделі та візуалізація
        r2_score(S_AV_Detect_sliding_wind, Yout_SV_AV_Detect_sliding_wind, 'MNK ЕКСПОНЕНТА_модель_згладжування')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
                'MNK ЕКСПОНЕНТА: Вибірка очищена від АВ алгоритм sliding_wind', path_plot + '7_LSM_expo.png')

    if (mode == 8):
        print('Регресія ЕКСПОНЕНТА')
        #  Очищення від аномальних похибок sliding window
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
        koef = mt.ceil(n * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки
        S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', filename_hist)
        Yout_SV_AV_Detect_sliding_wind = Expo_Regres(S_AV_Detect_sliding_wind, 10)
        Stat_characteristics_out_expo(SV_AV, Yout_SV_AV_Detect_sliding_wind,
                             'Регресія ЕКСПОНЕНТА, вибірка очищена від АВ алгоритм sliding_wind')
        # Оцінювання якості моделі та візуалізація
        r2_score_expo(S_AV_Detect_sliding_wind, Yout_SV_AV_Detect_sliding_wind, 'Регресія ЕКСПОНЕНТА_модель_згладжування')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
                'Регресія ЕКСПОНЕНТА: Вибірка очищена від АВ алгоритм sliding_wind', path_plot + '8_expo_regres.png')

    if (mode == 9):
        print('Нейромережа')
        batch_size = 256
        epochs = 10
        n_neurons = [499, 312, 156, 64]
        #  Очищення від аномальних похибок sliding window
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        koef_Extrapol = 0.5 # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
        n = 0.5
        koef = mt.ceil(n * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки
        S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', filename_hist)
        Yout_nnet = main_nnet(batch_size, epochs, S_AV_Detect_sliding_wind, n_neurons)
        Stat_characteristics_out_expo(SV_AV, Yout_nnet, 'Нейромережа, вибірка очищена від АВ алгоритм sliding_wind')
        #  Оцінювання якості моделі та візуалізація
        r2_score_expo(S_AV_Detect_sliding_wind, Yout_nnet, 'Регресія ЕКСПОНЕНТА_модель_згладжування')
        # УВАГА!! На графіках новедено нормована оцінка від нейромережі
        Plot_AV(Yout_nnet, S_AV_Detect_sliding_wind,'Нейромережа: Вибірка очищена від АВ алгоритм sliding_wind', path_plot + '9_n_net.png')

    if (mode == 10):
        print('ARIMA')
        #  Очищення від аномальних похибок sliding window
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
        n = 0.5
        koef = mt.ceil(n * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки
        S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind', filename_hist)
        data_out = main_ARIMA(S_AV_Detect_sliding_wind, path_plot + '10_ARIMA.png')


