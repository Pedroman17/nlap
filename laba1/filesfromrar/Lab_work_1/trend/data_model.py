'''
Модуль генерації синтезованих / модельних даних
'''

import numpy as np
import math as mt
import matplotlib.pyplot as plt


# рівномірний закон розводілу номерів АВ в межах вибірки
def randomAM(n, iter, nAV):

    SAV = np.zeros((nAV))
    S = np.zeros((n))

    for i in range(n):
        S[i] = np.random.randint(0, iter)  # параметри закону задаются межами аргументу
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)

    # генерація номерів АВ за рівномірним законом
    for i in range(nAV):
        SAV[i] = mt.ceil(np.random.randint(1, iter))  # рівномірний розкид номерів АВ в межах вибірки розміром 0-iter
    print('номери АВ: SAV=', SAV)
    print('----- статистичны характеристики РІВНОМІРНОГО закону розподілу ВВ -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------------------------')

    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()

    return SAV

# нормальний закон розводілу ВВ
def randoNORM(dm, dsig, iter):

    S = np.random.normal(dm, dsig, iter)  # нормальний закон розподілу ВВ з вибіркою єбємом iter та параметрами: dm, dsig
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)

    print('------- статистичны характеристики НОРМАЛЬНОЇ похибки вимірів -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('------------------------------------------------------------------')

    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()

    return S

#  модель ідеального тренду
def Model(n):

    S0=np.zeros((n))

    for i in range(n):
        S0[i] = (0.0000005*i*i)    # квадратична модель реального процесу

    return S0

# модель виміру (тренду) з нормальний шумом
def Model_NORM(SN, S0N, n):

    SV=np.zeros((n))

    for i in range(n):
        SV[i] = S0N[i]+SN[i]

    return SV

# модель виміру (тренду) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
def Model_NORM_AV(S0, SV, nAV, Q_AV, dm,  dsig, SAV):

    SV_AV = SV
    SSAV = np.random.normal(dm, (Q_AV * dsig), nAV)  # аномальна випадкова похибка з нормальним законом

    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]        # аномальні вимірів з рівномірно розподіленими номерами

    return SV_AV
