'''
Мордуль парсингу даних
'''

import pandas as pd
import numpy as np

def file_parsing (URL, File_name, Data_name):

    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
    S_real = np.zeros((len(values)))

    for i in range(len(values)):
        S_real[i] = values[i]

    print('Джерело даних: ', URL)

    return S_real
