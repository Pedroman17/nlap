import requests
from bs4 import BeautifulSoup
import csv
import re
from datetime import datetime
from urllib.parse import urljoin


BASE_URL = "https://www.pravda.com.ua"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def clean_text(text: str) -> str:
    """Очищення тексту: прибирає зайві пробіли/переноси/табуляції."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_html(url: str) -> str | None:
    """Надсилання HTTP-запиту + Отримання HTML-документа + перевірка відповіді."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
    except requests.RequestException:
        print("Вивести повідомлення про помилку: не вдалося виконати HTTP-запит.")
        return None

    if resp.status_code != 200:
        print(
            f"Вивести повідомлення про помилку: сервер повернув статус {resp.status_code}.")
        return None

    return resp.text


def extract_article_text(article_url: str) -> str:
    """Витяг повного тексту зі сторінки новини (p-теги)."""
    html = fetch_html(article_url)
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    text = " ".join(clean_text(p.get_text(" ", strip=True))
                    for p in paragraphs)
    text = clean_text(text)

    # фільтрація від пустих сторінок
    if len(text) < 200:
        return ""

    return text


def parse_article(link_tag, section_name: str, url: str) -> dict:
    """
    Обробка кожної новини:
    - заголовок + URL беремо зі списку
    - дату пробуємо знайти поруч у DOM (time)
    - опис = перші 200 символів повного тексту
    """
    title = clean_text(link_tag.get_text(" ", strip=True)) or "Без заголовка"

    time_text = "Дата не вказана"
    parent = link_tag.parent
    if parent:
        time_tag = parent.find("time")
        if time_tag:
            time_text = clean_text(time_tag.get_text(" ", strip=True))

    full_text = extract_article_text(url) if url else ""
    description = (full_text[:200] +
                   "...") if len(full_text) > 200 else full_text

    return {
        "section": section_name,
        "title": title,
        "time_text": time_text,
        "url": url,
        "description": description
    }


def is_good_link(full_url: str, link_text: str, section_url: str) -> bool:
    """Фільтр посилань: окремі правила для pravda.com.ua і life.pravda.com.ua."""
    if not full_url.startswith("http"):
        return False

    text = clean_text(link_text)
    if len(text) < 10:
        return False

    # відсіювання лишнього
    bad_parts = ["/tags/", "/search/", "/authors/", "#",
                 "facebook.com", "t.me", "twitter.com", "instagram.com"]
    if any(bp in full_url for bp in bad_parts):
        return False

    is_life = "life.pravda.com.ua" in section_url

    if not is_life:

        return ("/news/" in full_url) or ("/articles/" in full_url) or ("/columns/" in full_url)
    else:

        return "life.pravda.com.ua" in full_url


def parse_section(section_url: str, section_name: str, limit: int = 10) -> list:
    """
    Парсинг HTML → Пошук контейнерів/посилань → Цикл обробки кожної новини
    """
    print(f"\nЗавантаження розділу: {section_name}")

    html = fetch_html(section_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=True)

    seen = set()
    picked = []

    # Пошук посилань на новини
    for a in links:
        href = a.get("href", "").strip()
        if not href:
            continue

        if href.startswith("/"):
            full = urljoin(section_url, href)
        else:
            full = href

        if full.startswith("/"):
            full = urljoin(BASE_URL, full)

        if full in seen:
            continue

        if not is_good_link(full, a.get_text(" ", strip=True), section_url):
            continue

        seen.add(full)
        picked.append((a, full))

        if len(picked) >= limit:
            break

    news_list = []

    # тут відбувається обробка кожної новини
    for link_tag, url in picked:
        parsed = parse_article(link_tag, section_name, url)
        if parsed.get("description"):  # якщо вдалося витягнути хоча б якийсь текст
            news_list.append(parsed)

    print(f"Знайдено новин: {len(news_list)}")
    return news_list


def save_to_csv(data: list, filename: str) -> None:
    """Збереження у файл CSV (етап перед 'Кінець')."""
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Розділ", "Заголовок", "Дата", "Посилання", "Опис"])

        for item in data:
            writer.writerow([
                item.get("section", ""),
                item.get("title", ""),
                item.get("time_text", ""),
                item.get("url", ""),
                item.get("description", "")
            ])

    print(f"Збережено у файл: {filename}")


def main():

    print("ПАРСИНГ PRAVDA.COM.UA")

    # Ініціалізація параметрів (список розділів)
    available_sections = [
        {"name": "Новини", "url": "https://www.pravda.com.ua/news/"},
        {"name": "Політика", "url": "https://www.pravda.com.ua/news/politics/"},
        {"name": "Економіка", "url": "https://www.pravda.com.ua/news/economics/"},
        {"name": "Європейська правда", "url": "https://www.eurointegration.com.ua/"},
        {"name": "Життя", "url": "https://life.pravda.com.ua/"},
        {"name": "Колонки", "url": "https://www.pravda.com.ua/columns/"},
    ]

    print("\nДоступні розділи:")
    for i, section in enumerate(available_sections, start=1):
        print(f"{i}. {section['name']}")

    # Вибір 3 розділів
    while True:
        try:
            choices = input(
                "\nВиберіть 3 розділи (через кому, наприклад 1,2,3): ")
            indices = [int(x.strip()) - 1 for x in choices.split(",")]

            if len(indices) != 3:
                raise ValueError("Потрібно обрати рівно 3 розділи.")

            selected_sections = [available_sections[i] for i in indices]
            break
        except (ValueError, IndexError):
            print("Неправильний ввід. Спробуйте ще раз.")

    print("\nОбрані розділи:")
    for s in selected_sections:
        print(f"• {s['name']}")

    all_news = []

    for section in selected_sections:
        news = parse_section(section["url"], section["name"], limit=10)
        all_news.extend(news)

    # Якщо новини нема то  повідомлення і завершення
    if not all_news:
        print("\nНовини не знайдені.")
        return

    print(f"\nВсього зібрано: {len(all_news)} новин")

    # Збереження у файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pravda_news_{timestamp}.csv"
    save_to_csv(all_news, filename)

    print("\nГотово!")


if __name__ == "__main__":
    main()
