import requests
from bs4 import BeautifulSoup
import re
import csv
import pandas as pd
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

REGION = "Львів"

URLS = {
    "DIM.RIA": [
        ("https://dom.ria.com/uk/prodazha-kvartir/lvov/?rooms=3", "3-кімнатні квартири"),
        ("https://dom.ria.com/uk/prodazha-domov/lvov/", "Будинки"),
        ("https://dom.ria.com/uk/prodazha-domov/lvov/?type=duplex",
         "Дуплекси - первинний ринок")
    ],
    "OLX": [
        ("https://www.olx.ua/uk/nedvizhimost/kvartiry/lvov/?search%5Bfilter_enum_rooms%5D%5B0%5D=three",
         "3-кімнатні квартири"),
        ("https://www.olx.ua/uk/nedvizhimost/doma/lvov/", "Будинки"),
        ("https://www.olx.ua/uk/nedvizhimost/doma/lvov/q-%D0%B4%D1%83%D0%BF%D0%BB%D0%B5%D0%BA%D1%81/",
         "Дуплекси - первинний ринок")
    ],
    "Flatfy": [
        ("https://flatfy.ua/uk/search?query=3-%D0%BA%D1%96%D0%BC%D0%BD%D0%B0%D1%82%D0%BD%D0%B0%20%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D0%B0%20%D0%BB%D1%8C%D0%B2%D1%96%D0%B2", "3-кімнатні квартири"),
        ("https://flatfy.ua/uk/search?query=%D0%B1%D1%83%D0%B4%D0%B8%D0%BD%D0%BE%D0%BA%20%D0%BB%D1%8C%D0%B2%D1%96%D0%B2", "Будинки"),
        ("https://flatfy.ua/uk/search?query=%D0%B4%D1%83%D0%BF%D0%BB%D0%B5%D0%BA%D1%81%20%D0%BB%D1%8C%D0%B2%D1%96%D0%B2",
         "Дуплекси - первинний ринок")
    ]
}


def fetch_html(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Сторінка пропущена ({response.status_code}): {url}")
            return None
        return response.text
    except requests.RequestException:
        print(f"Помилка завантаження: {url}")
        return None


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_price(text):
    if not text:
        return None

    text = text.replace("\xa0", " ").replace(",", ".")
    matches = re.findall(r"[\d\s]+", text)
    for match in matches:
        digits = re.sub(r"\D", "", match)
        if digits.isdigit():
            value = int(digits)
            if 1000 <= value <= 100000000:
                return value
    return None


def parse_cards(html, platform, category):
    soup = BeautifulSoup(html, "html.parser")
    items = []

    selectors = [
        "article",
        "div[data-cy='l-card']",
        "div.realty-item",
        "div.css-1sw7q4x",
        "li",
        "div"
    ]

    cards = []
    for selector in selectors:
        found = soup.select(selector)
        useful = [x for x in found if len(x.get_text(" ", strip=True)) > 40]
        if len(useful) >= 5:
            cards = useful
            break

    for card in cards[:100]:
        text = clean_text(card.get_text(" ", strip=True))
        if len(text) < 40:
            continue

        title = ""
        title_tag = card.find(["h1", "h2", "h3", "h4"]) or card.find("a")
        if title_tag:
            title = clean_text(title_tag.get_text(" ", strip=True))

        price = extract_price(text)
        location = REGION if REGION.lower() in text.lower() else ""

        items.append({
            "platform": platform,
            "category": category,
            "title": title,
            "description": text,
            "price": price,
            "location": location
        })

    return items


def scrape_all():
    all_data = []

    for platform, pages in URLS.items():
        for url, category in pages:
            print(f"\nЗавантаження: {platform} -> {url}")
            html = fetch_html(url)
            if html is None:
                continue

            items = parse_cards(html, platform, category)
            all_data.extend(items)

    return all_data


def filter_data(data):
    filtered = []

    for item in data:
        if item["price"] is None:
            continue
        if item["price"] < 1000 or item["price"] > 100000000:
            continue
        filtered.append(item)

    return filtered


def aggregate_stats(data):
    grouped = {}

    for item in data:
        category = item["category"]
        if category not in grouped:
            grouped[category] = []

        grouped[category].append(item["price"])

    results = []

    for category, prices in grouped.items():
        results.append({
            "category": category,
            "count": len(prices),
            "avg_price": round(mean(prices), 2),
            "min_price": min(prices),
            "max_price": max(prices)
        })

    return results


def save_raw_data(data, filename="real_estate_raw.csv"):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Платформа", "Категорія",
                        "Назва", "Опис", "Ціна", "Регіон"])

        for item in data:
            writer.writerow([
                item["platform"],
                item["category"],
                item["title"],
                item["description"],
                item["price"],
                item["location"]
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


def run_neural_network(data):
    df = pd.DataFrame(data)

    if len(df) < 15:
        return None, "Недостатньо даних для нейромережевого аналізу"

    if df["category"].nunique() < 2:
        return None, "Недостатньо категорій для нейромережевого аналізу"

    counts = df["category"].value_counts()
    if counts.min() < 2:
        return None, "Недостатньо прикладів у кожній категорії для нейромережевого аналізу"

    X = df[["price"]]
    y = df["category"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = MLPClassifier(hidden_layer_sizes=(
            10, 10), max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        return round(accuracy * 100, 2), None
    except Exception:
        return None, "Помилка під час нейромережевого аналізу"


def save_report(stats, nn_accuracy, nn_error, filename="analysis_report.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("АНАЛІЗ ПРОПОЗИЦІЙ НЕРУХОМОСТІ\n")
        f.write(f"Регіон: {REGION}\n\n")

        for row in stats:
            f.write(f"Категорія: {row['category']}\n")
            f.write(f"Кількість пропозицій: {row['count']}\n")
            f.write(f"Середня ціна: {row['avg_price']}\n")
            f.write(f"Мінімальна ціна: {row['min_price']}\n")
            f.write(f"Максимальна ціна: {row['max_price']}\n")
            f.write("-" * 40 + "\n")

        f.write("\nНейромережевий аналіз:\n")
        if nn_accuracy is not None:
            f.write(f"Точність моделі: {nn_accuracy}%\n")
        else:
            f.write(f"{nn_error}\n")


def main():
    print("=== АНАЛІЗ ПРОПОЗИЦІЙ НЕРУХОМОСТІ ===")

    raw_data = scrape_all()
    print(f"\nУсього зібрано записів: {len(raw_data)}")

    filtered_data = filter_data(raw_data)
    print(f"Після фільтрації залишилось: {len(filtered_data)}")

    stats = aggregate_stats(filtered_data)

    print("\nПідсумкова статистика:")
    for row in stats:
        print(row)

    nn_accuracy, nn_error = run_neural_network(filtered_data)

    if nn_accuracy is not None:
        print(f"\nТочність нейромережі: {nn_accuracy}%")
    else:
        print(f"\nНейромережевий аналіз не виконано: {nn_error}")

    save_raw_data(filtered_data)
    save_stats(stats)
    save_report(stats, nn_accuracy, nn_error)

    print("\nФайли збережено:")
    print(" - real_estate_raw.csv")
    print(" - real_estate_stats.csv")
    print(" - analysis_report.txt")


if __name__ == "__main__":
    main()
