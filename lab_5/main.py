import requests
from bs4 import BeautifulSoup
import re
import csv
from statistics import mean

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

REGION = "Львів"


URLS = {
    "DIM.RIA": [
        # 3-кімнатні квартири
        "https://dom.ria.com/uk/prodazha-kvartir/lvov/?rooms=3",

        # будинки
        "https://dom.ria.com/uk/prodazha-domov/lvov/",

        # дуплекси
        "https://dom.ria.com/uk/prodazha-domov/lvov/?type=duplex"
    ],

    "OLX": [
        # 3-кімнатні квартири
        "https://www.olx.ua/uk/nedvizhimost/kvartiry/lvov/?search%5Bfilter_enum_rooms%5D%5B0%5D=three",

        # будинки
        "https://www.olx.ua/uk/nedvizhimost/doma/lvov/",

        # дуплекси
        "https://www.olx.ua/uk/nedvizhimost/doma/lvov/q-%D0%B4%D1%83%D0%BF%D0%BB%D0%B5%D0%BA%D1%81/"
    ],

    "Flatfy": [
        # квартири
        "https://flatfy.ua/uk/%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D0%B8-%D0%BB%D1%8C%D0%B2%D1%96%D0%B2",

        # будинки
        "https://flatfy.ua/uk/%D0%B1%D1%83%D0%B4%D0%B8%D0%BD%D0%BA%D0%B8-%D0%BB%D1%8C%D0%B2%D1%96%D0%B2",

        # дуплекси
        "https://flatfy.ua/uk/search?query=%D0%B4%D1%83%D0%BF%D0%BB%D0%B5%D0%BA%D1%81+%D0%BB%D1%8C%D0%B2%D1%96%D0%B2"
    ]
}


def fetch_html(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Помилка завантаження {url}: {e}")
        return None


def extract_price(text):
    if not text:
        return None

    text = text.replace(" ", "").replace("\xa0", "")
    match = re.search(r"(\d[\d\s]*)", text)
    if match:
        number = re.sub(r"[^\d]", "", match.group(1))
        if number.isdigit():
            return int(number)
    return None


def classify_category(title, description):
    full_text = f"{title} {description}".lower()

    if "3-кім" in full_text or "3 кім" in full_text or "3к" in full_text:
        return "3-кімнатні квартири"

    if "дуплекс" in full_text:
        if (
            "первин" in full_text
            or "новобуд" in full_text
            or "від забудовника" in full_text
            or "здача" in full_text
            or "новий" in full_text
        ):
            return "Дуплекси - первинний ринок"
        return "Дуплекси"

    if "будинок" in full_text or "house" in full_text or "котедж" in full_text:
        return "Будинки"

    return "Інше"


def parse_cards_generic(html, platform):
    soup = BeautifulSoup(html, "html.parser")
    items = []

    # набір можливих контейнерів
    card_selectors = [
        "article",
        "div[data-cy='l-card']",
        "div.realty-item",
        "div.css-1sw7q4x",
        "div[data-testid='listing-grid'] div",
        "section div"
    ]

    cards = []
    for selector in card_selectors:
        found = soup.select(selector)
        if len(found) >= 5:
            cards = found
            break

    for card in cards[:80]:
        text = card.get_text(" ", strip=True)
        if len(text) < 30:
            continue

        # заголовок
        title = ""
        title_tag = (
            card.find(["h1", "h2", "h3", "h4"])
            or card.find("a")
        )
        if title_tag:
            title = title_tag.get_text(" ", strip=True)

        # опис
        description = text

        # ціна
        price = extract_price(text)

        # регіон
        location = ""
        if REGION.lower() in text.lower():
            location = REGION

        category = classify_category(title, description)

        items.append({
            "platform": platform,
            "title": title,
            "description": description,
            "price": price,
            "location": location,
            "category": category
        })

    return items


def scrape_platform(platform_name, urls):
    all_items = []

    for url in urls:
        print(f"\nЗавантаження: {platform_name} -> {url}")
        html = fetch_html(url)
        if not html:
            continue

        items = parse_cards_generic(html, platform_name)
        all_items.extend(items)

    return all_items


def filter_needed_categories(data):
    allowed = {
        "3-кімнатні квартири",
        "Будинки",
        "Дуплекси - первинний ринок"
    }

    result = []
    for item in data:
        if item["category"] in allowed:
            result.append(item)
    return result


def aggregate_by_category(data):
    stats = {}

    for item in data:
        category = item["category"]
        if category not in stats:
            stats[category] = {
                "count": 0,
                "prices": []
            }

        stats[category]["count"] += 1

        if item["price"] is not None:
            stats[category]["prices"].append(item["price"])

    final_stats = []

    for category, values in stats.items():
        prices = values["prices"]

        final_stats.append({
            "category": category,
            "count": values["count"],
            "avg_price": round(mean(prices), 2) if prices else 0,
            "min_price": min(prices) if prices else 0,
            "max_price": max(prices) if prices else 0
        })

    return final_stats


def save_raw_data(data, filename="real_estate_raw.csv"):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Платформа", "Назва", "Опис",
                        "Ціна", "Регіон", "Категорія"])

        for item in data:
            writer.writerow([
                item["platform"],
                item["title"],
                item["description"],
                item["price"],
                item["location"],
                item["category"]
            ])


def save_stats(stats, filename="real_estate_stats.csv"):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Категорія", "Кількість", "Середня ціна",
                        "Мінімальна ціна", "Максимальна ціна"])

        for row in stats:
            writer.writerow([
                row["category"],
                row["count"],
                row["avg_price"],
                row["min_price"],
                row["max_price"]
            ])


def save_report(stats, filename="analysis_report.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Порівняльний аналіз пропозицій нерухомості\n")
        f.write(f"Регіон: {REGION}\n\n")

        for row in stats:
            f.write(f"Категорія: {row['category']}\n")
            f.write(f"Кількість пропозицій: {row['count']}\n")
            f.write(f"Середня ціна: {row['avg_price']}\n")
            f.write(f"Мінімальна ціна: {row['min_price']}\n")
            f.write(f"Максимальна ціна: {row['max_price']}\n")
            f.write("-" * 40 + "\n")


def main():
    print("=== АНАЛІЗ ПРОПОЗИЦІЙ НЕРУХОМОСТІ ===")

    all_data = []

    for platform_name, urls in URLS.items():
        platform_data = scrape_platform(platform_name, urls)
        all_data.extend(platform_data)

    print(f"\nУсього зібрано записів: {len(all_data)}")

    filtered_data = filter_needed_categories(all_data)
    print(f"Після фільтрації залишилось: {len(filtered_data)}")

    stats = aggregate_by_category(filtered_data)

    print("\nПідсумкова статистика:")
    for row in stats:
        print(row)

    save_raw_data(filtered_data)
    save_stats(stats)
    save_report(stats)

    print("\nФайли збережено:")
    print(" - real_estate_raw.csv")
    print(" - real_estate_stats.csv")
    print(" - analysis_report.txt")


if __name__ == "__main__":
    main()
