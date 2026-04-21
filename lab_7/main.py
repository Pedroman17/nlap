import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from deep_translator import GoogleTranslator
import re
import time

BASE_URL = "https://www.pravda.com.ua/news/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_page_url(page_number):
    if page_number == 1:
        return BASE_URL
    return f"https://www.pravda.com.ua/news/page_{page_number}/"


def parse_news_from_page(url):
    news_items = []

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Помилка завантаження сторінки {url}: {e}")
        return news_items

    soup = BeautifulSoup(response.text, "html.parser")

    # Спроба знайти новинні заголовки
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "a"]):
        text = clean_text(tag.get_text(" ", strip=True))

        if len(text) < 20:
            continue

        href = ""
        if tag.name == "a":
            href = tag.get("href", "")
        else:
            a = tag.find("a")
            if a:
                href = a.get("href", "")

        if "/news/" in href or "pravda.com.ua" in href or text:
            news_items.append(text)

    # прибираємо дублікати
    unique_items = []
    seen = set()

    for item in news_items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return unique_items


def collect_news(min_count=50):
    all_news = []
    seen = set()
    page = 1

    while len(all_news) < min_count and page <= 10:
        url = get_page_url(page)
        print(f"Парсинг сторінки: {url}")

        page_news = parse_news_from_page(url)

        for item in page_news:
            if item not in seen:
                seen.add(item)
                all_news.append(item)

        page += 1
        time.sleep(1)

    return all_news


def save_news_to_file(news_list, filename="parsed_news.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, news in enumerate(news_list, start=1):
            f.write(f"{i}. {news}\n")


def translate_news(news_list, target_lang="en"):
    translated = []

    translator = GoogleTranslator(source="auto", target=target_lang)

    for item in news_list:
        try:
            translated_text = translator.translate(item)
            translated.append(translated_text)
        except Exception:
            translated.append(item)

    return translated


def text_to_speech(news_list, lang="uk", output_file="news_audio.mp3"):
    full_text = ". ".join(news_list)

    tts = gTTS(text=full_text, lang=lang)
    tts.save(output_file)


def main():
    print("=== ОЗВУЧЕННЯ СТРІЧКИ НОВИН ===")

    news = collect_news(min_count=50)

    print(f"\nЗібрано новин: {len(news)}")

    save_news_to_file(news, "parsed_news.txt")
    print("Новини збережено у файл parsed_news.txt")

    lang_choice = input("Оберіть мову озвучування (uk/en): ").strip().lower()

    if lang_choice == "en":
        print("Переклад новин англійською...")
        news_for_audio = translate_news(news, target_lang="en")
        text_to_speech(news_for_audio, lang="en",
                       output_file="news_audio_en.mp3")
        print("Озвучення збережено у файл news_audio_en.mp3")

    else:
        text_to_speech(news, lang="uk", output_file="news_audio_uk.mp3")
        print("Озвучення збережено у файл news_audio_uk.mp3")


if __name__ == "__main__":
    main()
