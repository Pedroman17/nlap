
'''

Приклад парсінгу стрічки новин
'''


from bs4 import BeautifulSoup
import requests


# ----------------------------- Парсер сайту новин rdk  --------------------------------------
def Parser_URL_rdk (url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')        # аналіз структури html документу
    print(soup)
    quotes = soup.find_all('div', class_='newsline')   # вилучення із html документу ленти новин

    with open('test_1.txt', "w", encoding="utf-8") as output_file:
        print('----------------------- Лента новин', url, '---------------------------------')
        for quote in quotes:
            print(quote.text)
            output_file.write(quote.text)  # запис ленти новин до текстового файлу
        print('------------------------------------------------------------------------------')

    return

# ----------------------------- Парсер сайту новин pressorg24 ----------------------------------
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


if __name__ == '__main__':

    print('Оберіть напрям досліджень:')
    print('1 - Парсинг сайту новин https://www.rbc.ua/rus/news')
    print('2 - Парсинг сайту новин http://pressorg24.com/news')
    mode = int(input('mode:'))

    if (mode == 1):
        # ----------------- ПРИКЛАД парсингу_1 сайтів новин метод: GET -------------------------
        print('Джерело: https://www.rbc.ua')
        url = 'https://www.rbc.ua'
        Parser_URL_rdk(url)

    if (mode == 2):
        # ----------------- ПРИКЛАД парсингу_2 сайтів новин метод: GET -------------------------
        print('Обрано інформаційне джерело: http://pressorg24.com/news')
        url = 'http://pressorg24.com/news'
        Parser_URL_pressorg24(url)

