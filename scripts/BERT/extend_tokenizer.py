from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from transformers import AutoTokenizer
import datasets


def find_common_words(texts):
    corpus = [word for text in texts for word in text.split(" ")]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    freqs = X.sum(axis=0).tolist()[0]
    hist = np.histogram(freqs, bins=100)
    threshold = int(hist[1][1])
    n = hist[1][np.where(hist[1] >= threshold)[0][0]]
    counts_n = [x for x, y in zip(vectorizer.get_feature_names(), freqs) if y > n]
    return counts_n


def extend_tokenizer(tokenizer, texts):
    added_tokens = tokenizer.add_tokens(find_common_words(texts))
    print(f"Added {added_tokens} new tokens")
    return tokenizer


roberta_base_path = "roberta-base"
bert_base_path = (
    "bert-base-multilingual-cased"
)


tokenizer = AutoTokenizer.from_pretrained(bert_base_path)
dataset = datasets.load_from_disk(
    "data/dataset"
)

tokenizer = extend_tokenizer(tokenizer, dataset["train"]["pre_text"])
texts = dataset["train"]["pre_text"]
tokenizer.save_pretrained("ecthr/bert-base-ecthr-ex")
