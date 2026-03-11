# -------------- Технології text mining за даними парсінгу сайтів ------------

'''
Технології text mining за даними парсінгу сайтів: частотний аналіз текстових повідомлень


Package                      Version
---------------------------- -----------
bs4                          0.0.1
matplotlib                   3.6.2
requests                     2.28.2
wordcloud                    1.8.2.2

'''


import re                                                           # підтримка регулярних виразів
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


# ---------- Парсер САЙТУ для отримання html структури і вилучення з неї стрічкі новин  --------
def Parser_URL_rdk (url):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'lxml')        # аналіз структури html документу
    print(soup)
    quotes = soup.find_all('div', class_='newsline')   # вилучення із html документу ленти новин

    with open('test_2.txt', "w", encoding="utf-8") as output_file:
        print('----------------------- Лента новин', url, '---------------------------------')
        for quote in quotes:
            print(quote.text)
            output_file.write(quote.text)  # запис ленти новин до текстового файлу
        print('------------------------------------------------------------------------------')

    return

# ---------- Парсер САЙТУ для отримання html структури і вилучення з неї стрічкі новин  --------
def Parser_URL_pressorg24 (url):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'lxml')        # аналіз структури html документу
    print(soup)
    quotes = soup.find_all('div', class_='event')   # вілучення із html документу ленти новин

    with open('test_2.txt', "w", encoding="utf-8") as output_file:
        print('----------------------- Лента новин', url, '---------------------------------')
        for quote in quotes:
            print(quote.text)
            output_file.write(quote.text)  # запис ленти новин до текстового файлу
        print('------------------------------------------------------------------------------')

    return

# -------------- Частотний text mining аналіз даних, від сайтів новин ---------------------
def text_mining_wordcloud(f):
    text = str(f.readlines())
    print(text)
    # -------------- Аналіз тексту на частоту слів ----------------
    words = re.findall('[a-zA-Z]{2,}', text)    # Регулярний вираз для слів - більше 2 букв - англомовний контент
    stats = {}
    print(words)
    for w in words:
        stats[w] = stats.get(w, 0) + 1

    # -------------- Виявлення токенів у тексті ----------------
    w_ranks = sorted(stats.items(), key=lambda x: x[1], \
                     reverse=True)[0:10]

    _wrex = re.findall('[a-zA-Z]+', str(w_ranks))     # Регулярний вираз для слів - англомовний контент
    _drex = re.findall('[0-9]+', str(w_ranks))

    pl = [p for p in range(1, 11)]
    for j in range(len(_wrex)):
        places = '{} place,{} - {} times'.format(pl[j], _wrex[j], _drex[j])
        print(places)

    text_raw = " ".join(_wrex)  # перетворення токінів у строку

    # ----------------- Побудова домінантної хмари  --------------
    wordcloud = WordCloud().generate(text_raw)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    return

# ---------- Докладний частотний text mining аналіз даних від сайтів новин --------------
def text_mining_kr(filename):
    # ----------------- реалізовано на рівні роботи із рядком  --------------
    with open(filename, encoding="utf-8") as file:
        text = file.read()
    text = text.replace("\n", " ")
    text = text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    text = text.lower()
    words = text.split()
    words.sort()
    words_dict = dict()
    for word in words:
        if word in words_dict:
            words_dict[word] = words_dict[word] + 1
        else:
            words_dict[word] = 1
    print("Кількість слів: %d" % len(words))
    print("Кількість унікальних слів: %d" % len(words_dict))
    print("Усі використані слова:")
    for word in words_dict:
        print(word.ljust(20), words_dict[word])
    return


# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ------------------------------
if __name__ == '__main__':
    print('Оберіть інформаційне джерело:')
    # -------------- Головні виклики парсера для отримання даних text mining --------------------
    print('1 - https://www.rbc.ua/rus/news', 'text mining')
    print('2 - http://pressorg24.com/news', 'text mining')
    mode = int(input('mode:'))

    if (mode == 1):
        print('Обрано інформаційне джерело: https://www.rbc.ua/rus/news')
        url = 'https://www.rbc.ua/rus/news'
        Parser_URL_rdk(url)
        # -------------- Частотний text mining аналіз даних від сайтів новин --------------------
        f = open('test_2.txt', 'r', encoding="utf-8")
        print('Домінуючий контент сайту:', mode, ':', url)
        text_mining_wordcloud(f)
        print('Докладний частотний аналіз інформаційного джерела:', mode, ':', url)
        filename = 'test_2.txt'
        text_mining_kr(filename)

    if (mode == 2):
        print('Обрано інформаційне джерело: http://pressorg24.com/news')
        url = 'http://pressorg24.com/news'
        Parser_URL_pressorg24(url)
        # -------------- Частотний text mining аналіз даних від сайтів новин --------------------
        f = open('test_2.txt', 'r', encoding="utf-8")
        print('Домінуючий контент сайту:', mode, ':', url)
        text_mining_wordcloud(f)
        print('Докладний частотний аналіз інформаційного джерела:', mode, ':', url)
        filename = 'test_2.txt'
        text_mining_kr(filename)

