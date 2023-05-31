from transformers import AutoTokenizer
import numpy as np
import datasets
import statistics


data = datasets.load_from_disk("data/ecthr_cases_violated")
data.cleanup_cache_files()
model_path = "roberta-base-ecthr"

tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize(example):
    tokens = tokenizer(example["facts"], return_tensors="np")
    example["input_ids"] = tokens["input_ids"]
    example["num_tokens"] = [len(x) for x in tokens["input_ids"]]
    example["attention_mask"] = tokens["attention_mask"]
    return example


MAX_LENGTH = 4096
data_ = data.map(tokenize)
num_tokens = [sum(x) for x in data_["train"]["num_tokens"]]
print(model_path)
print(f"vocab_size tokenizer: {len(tokenizer)}")
print(f"quantiles num tokens stacked facts: {statistics.quantiles(num_tokens)}")
print(f"mean num tokens stacked facts: {statistics.mean(num_tokens)}")
print(
    f'unique_tokens: {len(set([z for x in data_["train"]["input_ids"] for y in x for z in set(y)]))}'
)
num_tokens_removed = [
    np.clip(np.array(x) - MAX_LENGTH, 0, None) for x in data_["train"]["num_tokens"]
]
print(
    f"avg tokens removed / fact: {statistics.mean([statistics.mean(x) for x in num_tokens_removed])}"
)
print(
    f"avg tokens removed / doc: {statistics.mean([sum(x) for x in num_tokens_removed])}"
)
print(f"total tokens removed: {sum([sum(x) for x in num_tokens_removed])}")
