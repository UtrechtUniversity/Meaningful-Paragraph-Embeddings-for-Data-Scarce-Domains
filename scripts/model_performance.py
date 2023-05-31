from ml_utils import gpu
from transformers import AutoModel, AutoTokenizer
import time

models = [
    "legal-bert-base-uncased",
    "bert-base-multilingual-cased",
    "bert-base-ecthr_tiny",
    "roberta-base",
    "longformer-base-4096",
]
from statistics import stdev, mean

for model_path in models:
    model = AutoModel.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    str = "hello my name is" * 4000

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

    result = gpu.estimate_batch_size_rec(encode, str, size=100000)
    time_res = []
    for i in range(10):
        print(i)
        st = time.process_time() * 1000
        encode(str)
        et = time.process_time() * 1000
        encode([str] * result)
        ett = time.process_time() * 1000
        time_res.append((et - st, ett - et))
    time_res_1, time_res_max = list(zip(*time_res))

    print(model_path)
    print(
        f"""
        hidden_size: {model.config.hidden_size}
        num_attention_heads: {model.config.num_attention_heads}
        num_hidden_layers/transformer blocks: {model.config.num_hidden_layers}
        max BS: {result}
        avg 1: {mean(time_res_1)} seconds ± {stdev(time_res_1)}
        avg max: {mean(time_res_max)} seconds ± {stdev(time_res_max)}


    """
    )
