
'''
Модуль статистичного навчання
'''


import numpy as np
import math as mt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


# Коефіцієнт детермінації - оцінювання якості полінома
def r2_score(SL, Yout, Text):

    # статистичні характеристики вибірки з урахуванням тренду
    iter = len(Yout)
    numerator = 0
    denominator_1 = 0

    for i in range(iter):
        numerator = numerator + (SL[i] - Yout[i, 0]) ** 2
        denominator_1 = denominator_1 + SL[i]

    denominator_2 = 0

    for i in range(iter):
        denominator_2 = denominator_2 + (SL[i] - (denominator_1 / iter)) ** 2

    R2_score_our = 1 - (numerator / denominator_2)
    print('------------', Text, '-------------')
    print('кількість елементів вбірки=', iter)
    print('Коефіцієнт детермінації (ймовірність апроксимації)=', R2_score_our)

    return R2_score_our



# Коефіцієнт детермінації - оцінювання якості експоненти
def r2_score_expo(SL, Yout, Text):

    # статистичні характеристики вибірки з урахуванням тренду
    iter = len(Yout)
    numerator = 0
    denominator_1 = 0

    for i in range(iter):
        numerator = numerator + (SL[i] - Yout[i]) ** 2
        denominator_1 = denominator_1 + SL[i]

    denominator_2 = 0

    for i in range(iter):
        denominator_2 = denominator_2 + (SL[i] - (denominator_1 / iter)) ** 2

    R2_score_our = 1 - (numerator / denominator_2)
    print('------------', Text, '-------------')
    print('кількість елементів вбірки=', iter)
    print('Коефіцієнт детермінації (ймовірність апроксимації)=', R2_score_our)

    return R2_score_our



# МНК згладжування
def LSM(S0):

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))

    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)

    print('Регресійна модель:')
    print('y(t) = ', C[0,0], ' + ', C[1,0], ' * t', ' + ', C[2,0], ' * t^2')

    return Yout



# МНК детекція та очищення АВ
def LSM_AV_Detect(S0):

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))

    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)

    return C[1, 0]



# МНК прогнозування
def LSM_Extrapol(S0, koef):

    iter = len(S0)
    Yout_Extrapol = np.zeros((iter+koef, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))

    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)

    print('Регресійна модель:')
    print('y(t) = ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2')

    for i in range(iter+koef):
        Yout_Extrapol[i, 0] = C[0, 0]+C[1, 0]*i+(C[2, 0]*i*i)   # проліноміальна крива МНК - прогнозування

    return Yout_Extrapol



#  МНК експонента
def LSM_exponent(S0):

    iter = len(S0)
    Yout = np.zeros((iter, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 4))

    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)

    c0 = C[0, 0]
    c1 = C[1, 0]
    c2 = C[2, 0]
    c3 = C[3, 0]
    a3 = 3 * (c3 / c2)
    a2 = (2*c2)/(a3**2)
    a0 = c0 - a2
    a1 = c1-(a2*a3)

    print('Регресійна модель:')
    print('y(t) = ', a0, ' + ', a1, ' * t', ' + ', a2, ' * exp(', a3, ' * t )')

    for i in range(iter):
        Yout[i, 0] = a0 + a1 * i + a2 * mt.exp(a3 * i)

    return Yout



# Expo_scipy
def Expo_Regres (Yin, bstart):

    def func_exp(x, a, b, c, d):
        print('Регресійна модель:')
        print('y(t) = ', c, ' + ', d, ' * t', ' + ', a, ' * exp(', b, ' * t)')
        return a * np.exp(b * x) + c + (d * x)

    # емпірічні коефіцієнти старта
    aStart=bstart/10
    bStart=bstart/1000
    cStart=bstart+10
    dStart=bstart/10
    iter = len(Yin)
    x_data = np.ones((iter))
    y_data = np.ones((iter))

    for i in range(iter):
        x_data[i] = i
        y_data[i] = Yin[i]


    popt, pcov = curve_fit(func_exp, x_data, y_data, p0=(aStart, bStart, cStart, dStart))

    return func_exp(x_data, *popt)



# алгоритм -а-b фільтрa
def ABF(S0):

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    YoutAB = np.zeros((iter, 1))
    T0 = 1

    for i in range(iter):
        Yin[i, 0] = float(S0[i])

    # початкові дані для запуску фільтра
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
    Yextra = Yin[0, 0] + Yspeed_retro
    alfa = 2 * (2 * 1 - 1) / (1 * (1 + 1))
    beta = (6 / 1) * (1 + 1)
    YoutAB[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])

    # рекурентний прохід по вимірам
    for i in range(1, iter):
        YoutAB[i, 0] = Yextra + alfa * (Yin[i, 0] - Yextra)
        Yspeed = Yspeed_retro + (beta/T0) * (Yin[i, 0] - Yextra)
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i,0] + Yspeed_retro
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))

    return YoutAB



# Виявлення АВ за алгоритмом medium
def Sliding_Window_AV_Detect_medium(S0, n_Wind, Q):

    # параметри циклів
    iter = len(S0)
    j_Wind=mt.ceil(iter-n_Wind)+1
    S0_Wind=np.zeros((n_Wind))

    # еталон
    j = 0
    for i in range(n_Wind):
        l = (j + i)
        S0_Wind[i] = S0[l]
        dS_standart = np.var(S0_Wind)
        scvS_standart = mt.sqrt(dS_standart)

    # ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l=(j+i)
            S0_Wind[i] = S0[l]

    # Стат хар ковзного вікна
        mS = np.median(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)

    # детекція та заміна АВ
        if scvS > (Q * scvS_standart):
            # детектор виявлення АВ
            S0[l] = mS

    return S0

#  Виявлення АВ за МНК
def Sliding_Window_AV_Detect_LSM (S0, Q, n_Wind):

    #  параметри циклів
    iter = len(S0)
    j_Wind=mt.ceil(iter-n_Wind)+1
    S0_Wind=np.zeros((n_Wind))

    # еталон
    Speed_standart = LSM_AV_Detect(S0)
    Yout_S0 = LSM(S0)

    #  ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l=(j+i)
            S0_Wind[i] = S0[l]

    #  Стат хар ковзного вікна
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)

    #  детекція та заміна АВ
        Speed_standart_1 = abs(Speed_standart * mt.sqrt(iter))
        Speed_1 = abs(Q * Speed_standart * mt.sqrt(n_Wind) * scvS)
        if Speed_1  > Speed_standart_1:
            # детектор виявлення АВ
            S0[l] = Yout_S0[l, 0]

    return S0



# ------------------------------ Виявлення АВ за алгоритмом sliding window -------------------------------------
def Sliding_Window_AV_Detect_sliding_wind (S0, n_Wind):

    # параметри циклів
    iter = len(S0)
    j_Wind=mt.ceil(iter-n_Wind)+1
    S0_Wind=np.zeros((n_Wind))
    Midi = np.zeros(( iter))

    #  ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l=(j+i)
            S0_Wind[i] = S0[l]
        # Стат хар ковзного вікна
        Midi[l] = np.median(S0_Wind)

    # очищена вибірка
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]

    for j in range(n_Wind):
        S0_Midi[j] = S0[j]

    return S0_Midi
