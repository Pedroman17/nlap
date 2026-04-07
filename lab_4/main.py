import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

URL = "https://huggingface.co/papers/trending"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_html(url):
    response = requests.get(url, headers=HEADERS)
    return response.text


def parse_reviews(html):
    soup = BeautifulSoup(html, "html.parser")

    titles = [tag.get_text(strip=True) for tag in soup.find_all("h3")]
    descriptions = []

    for tag in soup.find_all("h3"):
        p = tag.find_next("p")
        if p:
            descriptions.append(p.get_text(" ", strip=True))

    reviews = []
    for t, d in zip(titles, descriptions):
        reviews.append(t + " " + d)

    return reviews


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():

    print("=== LAB 4 NLP ===")

    # 1. Web scraping
    html = fetch_html(URL)
    reviews = parse_reviews(html)

    print("\nОтримано відгуків:", len(reviews))

    # беремо мінімум 5
    reviews = reviews[:10]

    # 2. очищення тексту
    cleaned_reviews = [clean_text(r) for r in reviews]

    # 3. створюємо позитивні і негативні мітки
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # 4. Bag of Words
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(cleaned_reviews)

    # 5. train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42
    )

    # 6. класифікація
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 7. прогноз
    predictions = model.predict(X_test)

    # 8. оцінка точності
    accuracy = accuracy_score(y_test, predictions)

    print("\nРезультати класифікації:")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nКласифікація тестових відгуків:")
    for i, text in enumerate(X_test):
        if predictions[i] == 1:
            print("Positive review")
        else:
            print("Negative review")


if __name__ == "__main__":
    main()
