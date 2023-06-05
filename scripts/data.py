from transformers import AutoModel, AutoTokenizer
import datasets

data = datasets.load_from_disk("/data/ecthr_cases_violated")
model_path = "bert-base-uncased"

model = AutoModel.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize(example):
    tokens = tokenizer(
        example["facts"],
        return_tensors="np",
        max_length=model.config.max_position_embeddings - 2,
        truncation=True,
        padding=True,
    )
    example["input_ids"] = tokens["input_ids"]
    example["attention_mask"] = tokens["attention_mask"]
    return example


def tokenize_vectorize(example):
    facts = example["facts"]
    vectors = []
    input_ids = []
    attention_masks = []

    for fact in batch(facts, 1):
        vector, input_id, attention_mask = encode_multiple(fact)
        vectors.extend(vector)
        input_ids.extend(input_id)
        attention_masks.extend(attention_mask)

    example["input_ids"] = input_ids
    example["vectors"] = vectors
    example["attention_mask"] = attention_masks
    return example


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def encode(str):
    tokens = tokenizer(
        str,
        return_tensors="pt",
        max_length=model.config.max_position_embeddings - 2,
        truncation=True,
        padding=True,
    )
    vectors = model(**tokens.to("cuda"))["pooler_output"].detach().cpu()
    return vectors, tokens["input_ids"], tokens["attention_mask"]


def encode_multiple(strs):
    tokens = tokenizer(
        strs,
        return_tensors="pt",
        max_length=model.config.max_position_embeddings - 2,
        truncation=True,
        padding=True,
    )
    vectors = model(**tokens.to("cuda"))["pooler_output"].detach().cpu()
    return list(vectors), list(tokens["input_ids"]), list(tokens["attention_mask"])


data_ = data.map(tokenize_vectorize, load_from_cache_file=False)
data_.save_to_disk("data/ecthr_cases_violated_bert_base_cased")
