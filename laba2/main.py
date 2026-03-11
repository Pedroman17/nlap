import requests
from bs4 import BeautifulSoup
import re
from collections import Counter

URL = "https://huggingface.co/papers/trending"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

STOP_WORDS = {
    "the", "and", "is", "in", "at", "of", "a", "an", "to", "for", "on", "with",
    "by", "this", "that", "it", "as", "are", "was", "were", "be", "been", "or",
    "from", "but", "not", "can", "could", "should", "would", "we", "they", "you",
    "he", "she", "them", "his", "her", "their", "our", "us", "about", "into",
    "than", "then", "there", "here", "also", "such", "these", "those", "using",
    "used", "use", "new", "show", "paper", "model", "models"
}


def fetch_html(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Помилка HTTP-запиту: {e}")
        return None


def save_text(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        if isinstance(content, list):
            f.write("\n".join(content))
        else:
            f.write(content)


def parse_huggingface_papers(html):
    soup = BeautifulSoup(html, "html.parser")

    titles = [tag.get_text(strip=True) for tag in soup.find_all("h3")]

    descriptions = []
    for tag in soup.find_all("h3"):
        next_p = tag.find_next("p")
        if next_p:
            descriptions.append(next_p.get_text(" ", strip=True))

    texts = []
    for title, desc in zip(titles, descriptions):
        texts.append(f"{title}. {desc}")

    return texts


def filter_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text):
    return text.lower()


def tokenize_text(text):
    return re.findall(r"\b[a-z]+\b", text)


def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOP_WORDS and len(word) > 2]


def simple_lemmatize(word):

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    return word


def lemmatize_tokens(tokens):
    return [simple_lemmatize(word) for word in tokens]


def simple_stem(word):

    suffixes = ["ing", "edly", "ed", "ly", "es", "s", "ment"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def stem_tokens(tokens):
    return [simple_stem(word) for word in tokens]


def get_top_words(tokens, top_n=10):
    counter = Counter(tokens)
    return counter.most_common(top_n)


def preview_text(text, length=300):
    """Короткий фрагмент тексту для консолі."""
    if len(text) <= length:
        return text
    return text[:length] + "..."


def preview_list(items, count=15):
    """Короткий фрагмент списку для консолі."""
    return ", ".join(items[:count])


def main():
    print("Обробка текстового простору з Hugging Face Papers")

    html = fetch_html(URL)
    if not html:
        print("Не вдалося отримати HTML-документ.")
        return

    save_text("huggingface_html.html", html)

    raw_texts = parse_huggingface_papers(html)
    full_text = " ".join(raw_texts)
    save_text("1_input_text.txt", full_text)

    print("\n1.1 Отримання текстового простору:")
    print(f"Кількість текстових фрагментів: {len(raw_texts)}")
    print("Фрагмент тексту:")
    print(preview_text(full_text))

    filtered_text = filter_text(full_text)
    save_text("2_filtered_text.txt", filtered_text)

    print("\n1.2 Фільтрація тексту:")
    print("Фрагмент після фільтрації:")
    print(preview_text(filtered_text))

    normalized_text = normalize_text(filtered_text)
    save_text("3_normalized_text.txt", normalized_text)

    print("\n1.3 Нормалізація тексту:")
    print("Фрагмент після нормалізації:")
    print(preview_text(normalized_text))

    tokens = tokenize_text(normalized_text)
    save_text("4_tokens.txt", tokens)

    print("\n1.4 Токенізація:")
    print(f"Кількість токенів: {len(tokens)}")
    print("Перші токени:")
    print(preview_list(tokens))

    tokens_no_stop = remove_stopwords(tokens)
    save_text("5_no_stopwords.txt", tokens_no_stop)

    print("\n1.5 Видалення стоп-слів:")
    print(
        f"Кількість токенів після видалення стоп-слів: {len(tokens_no_stop)}")
    print("Перші токени:")
    print(preview_list(tokens_no_stop))

    lemmatized_tokens = lemmatize_tokens(tokens_no_stop)
    save_text("6_lemmatized.txt", lemmatized_tokens)

    print("\n1.6 Лематизація:")
    print("Перші лематизовані токени:")
    print(preview_list(lemmatized_tokens))

    stemmed_tokens = stem_tokens(tokens_no_stop)
    save_text("7_stemmed.txt", stemmed_tokens)

    print("\n1.7 Стемінг:")
    print("Перші токени після стемінгу:")
    print(preview_list(stemmed_tokens))

    top_words = get_top_words(lemmatized_tokens, 10)
    top_words_lines = [f"{word}: {count}" for word, count in top_words]
    save_text("8_top_words.txt", top_words_lines)

    print("\n1.8 Топ-10 слів:")
    for i, (word, count) in enumerate(top_words, start=1):
        print(f"{i}. {word} — {count}")

    print("\nГотово. Усі етапи збережені у txt-файли.")


if __name__ == "__main__":
    main()
