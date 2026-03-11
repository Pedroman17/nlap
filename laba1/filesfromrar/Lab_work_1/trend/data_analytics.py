'''
Модуль аналітики / аналізу даних
'''

import numpy as np
import math as mt
import matplotlib.pyplot as plt

# статистичні характеристики вхідної вибірки
def Stat_characteristics_in(SL, Text, filename):

    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))

    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]

    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)

    print('------------', Text ,'-------------')
    print('кількість елементів вбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------')

    plt.hist(SL0, bins=20, facecolor="blue", alpha=0.5)
    plt.savefig(filename)
    plt.show()

    return

# статистичні характеристики лінії тренда
def Stat_characteristics_out(SL_in, SL, Text):

    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))

    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]

    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)

    # глобальне лінійне відхилення оцінки - динамічна похибка моделі
    Delta = 0

    for i in range(iter):
        Delta = Delta + abs(SL_in[i] - Yout[i, 0])

    Delta_average_Out = Delta / (iter + 1)

    print('------------', Text ,'-------------')
    print('кількість елементів ивбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('Динамічна похибка моделі=', Delta_average_Out)
    print('-----------------------------------------------------')

    return

# статистичні характеристики лінії тренда
def Stat_characteristics_out_expo(SL_in, SL, Text):

    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))

    for i in range(iter):
        SL0[i] = SL[i] - Yout[i]

    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)

    # глобальне лінійне відхилення оцінки - динамічна похибка моделі
    Delta = 0
    for i in range(iter):
        Delta = Delta + abs(SL_in[i] - Yout[i])

    Delta_average_Out = Delta / (iter + 1)

    print('------------', Text ,'-------------')
    print('кількість елементів ивбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('Динамічна похибка моделі=', Delta_average_Out)
    print('-----------------------------------------------------')

    return


# статистичні характеристики екстраполяції
def Stat_characteristics_extrapol(koef, SL, Text):

    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))

    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)

    #  довірчий інтервал прогнозованих значень за СКВ
    scvS_extrapol = scvS * koef

    print('------------', Text ,'-------------')
    print('кількість елементів ивбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('Довірчий інтервал прогнозованих значень за СКВ=', scvS_extrapol)
    print('-----------------------------------------------------')

    return


# МНК згладжуваннядля визначення стат. характеристик
def MNK_Stat_characteristics (S0):

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))

    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)

    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)

    return Yout

# графіки тренда, вимірів з нормальним шумом
def Plot_AV (S0_L, SV_L, Text, filename):

    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.savefig(filename)
    plt.show()

    return
