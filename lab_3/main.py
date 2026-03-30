import re
import math
from collections import Counter

# Вхідні тексти


text_en = """
Natural language processing is a field of artificial intelligence.
It studies interaction between computers and human language.
Machine learning models analyze and generate text.
"""

text_ua = """
Обробка природної мови є галуззю штучного інтелекту.
Вона вивчає взаємодію між комп'ютерами та людською мовою.
Алгоритми машинного навчання аналізують текст.
"""

# Очищення тексту


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Zа-яА-Яіїєґ\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Токенізація


def tokenize(text):

    tokens = re.findall(r"\b[a-zA-Zа-яА-Яіїєґ]+\b", text)

    return tokens


# Простий POS-tagging


def simple_pos_tag(tokens):

    tags = []

    for word in tokens:

        if word.endswith("ing") or word.endswith("ed"):
            pos = "VERB"

        elif word.endswith("ly"):
            pos = "ADV"

        elif word.endswith("tion") or word.endswith("ment"):
            pos = "NOUN"

        elif word.endswith("ous") or word.endswith("able"):
            pos = "ADJ"

        else:
            pos = "WORD"

        tags.append((word, pos))

    return tags


# Bag of Words

def bag_of_words(tokens):

    return Counter(tokens)


# TF-IDF


def tfidf(tokens_list):

    idf = {}
    all_tokens = set()

    for tokens in tokens_list:
        all_tokens.update(tokens)

    N = len(tokens_list)

    for token in all_tokens:
        count = sum(token in tokens for tokens in tokens_list)
        idf[token] = math.log(N / (1 + count))

    vectors = []

    for tokens in tokens_list:

        tf_counter = Counter(tokens)

        vector = {}

        for token in all_tokens:
            vector[token] = tf_counter[token] * idf[token]

        vectors.append(vector)

    return vectors


# Cosine similarity


def cosine_similarity(vec1, vec2):

    dot = sum(vec1[x] * vec2[x] for x in vec1)

    norm1 = math.sqrt(sum(vec1[x] ** 2 for x in vec1))
    norm2 = math.sqrt(sum(vec2[x] ** 2 for x in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)


# Основна функція


def main():

    # 1 Очищення
    clean_en = clean_text(text_en)
    clean_ua = clean_text(text_ua)

    print("Clean EN text:")
    print(clean_en)

    print("\nClean UA text:")
    print(clean_ua)

    # 2 Токенізація
    tokens_en = tokenize(clean_en)
    tokens_ua = tokenize(clean_ua)

    print("\nTokens EN:")
    print(tokens_en)

    print("\nTokens UA:")
    print(tokens_ua)

    # 3 POS tagging
    print("\nPOS tagging EN:")

    pos_en = simple_pos_tag(tokens_en)

    for word, tag in pos_en:
        print(word, "-", tag)

    print("\nPOS tagging UA:")

    pos_ua = simple_pos_tag(tokens_ua)

    for word, tag in pos_ua:
        print(word, "-", tag)

    # 4 Bag of Words
    bow_en = bag_of_words(tokens_en)
    bow_ua = bag_of_words(tokens_ua)

    print("\nBag of Words EN:")
    print(bow_en)

    print("\nBag of Words UA:")
    print(bow_ua)

    # 5 TF-IDF
    tfidf_vectors = tfidf([tokens_en, tokens_ua])

    print("\nTF-IDF vectors:")
    print(tfidf_vectors)

    # 6 Cosine similarity
    sim = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])

    print("\nCosine similarity between texts:", sim)

    print("\nProgram finished.")


if __name__ == "__main__":
    main()
