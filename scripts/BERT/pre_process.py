import pandas as pd
import datasets
import os
import shutil

TAG = "pre_text"
TMP_DATASET_LOC = "./temp_dataset"


def pre_process_data(
    data_file_path,
    traindatapath,
    tag,
    tokenizer,
    splits={"train": 0.8, "test": 0.2},
    max_seq_length=512,
):

    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)
        print(f"Creating dir {data_file_path}")
    config_file = "data_example/BERTDataset/config.json"
    BERTDataset_file = "data_example/BERTDataset/BERTDataset.py"
    shutil.copy(config_file, data_file_path + "/config.json")
    shutil.copy(
        BERTDataset_file, f"{data_file_path}/{data_file_path.split('/')[-1]}.py"
    )
    if isinstance(traindatapath, str):
        traindatapath = [traindatapath]
    for t in traindatapath:
        _pre_process_data(t, tag)

    print(f"Maximum sequence length: {max_seq_length}")
    if tokenizer.is_fast:
        tokenize_fast(tokenizer, data_file_path, max_seq_length, splits)
    else:
        tokenizer.enable_truncation(max_seq_length)
        assert tokenizer.truncation["max_length"] == max_seq_length
        tokenize(tokenizer, data_file_path, splits)
    shutil.rmtree(TMP_DATASET_LOC)

    return tokenizer


def _pre_process_data(traindatapath, tag):
    print(f"Preprocessing {traindatapath}...")
    if traindatapath.endswith("pickle") or traindatapath.endswith("pkl"):
        traindata = pd.read_pickle(traindatapath)
        traindata = traindata.reset_index(drop=False)
        traindata["next_sentence_label"] = traindata[tag].apply(
            lambda x: [1] + ((len(x) - 1) * [0])
        )
        next_sentence_labels = traindata.explode("next_sentence_label")[
            "next_sentence_label"
        ].reset_index(drop=True)
        sents = traindata.explode(tag).reset_index()
        sents["next_sentence_label"] = next_sentence_labels
        sents = sents.dropna()
        all_words = (
            traindata[tag]
            .apply(lambda y: [z for x in y for z in x.split(" ")])
            .explode()
        )
        all_words.dropna(inplace=True)
    elif traindatapath.endswith(".txt"):
        with open(traindatapath, "r") as data:
            tag_data = map(lambda x: x.rstrip(), data.readlines())
            sents = pd.DataFrame(zip(tag_data), columns=[tag])
    elif os.path.isdir(traindatapath):
        for filename in os.listdir(traindatapath):
            f = os.path.join(traindatapath, filename)
            _pre_process_data(f, tag)
        return
    else:
        raise Exception(f"File extention {traindatapath} can't be interpreted!")

    sents = sents[sents[tag].notna()]
    sents = sents.rename(columns={tag: TAG})
    dataSet = datasets.Dataset.from_pandas(sents)
    if os.path.exists(TMP_DATASET_LOC):
        shutil.move(TMP_DATASET_LOC, TMP_DATASET_LOC + "_temp")
        dataSet = datasets.concatenate_datasets(
            [dataSet, datasets.load_from_disk(TMP_DATASET_LOC + "_temp")]
        )
    dataSet.save_to_disk(TMP_DATASET_LOC)
    shutil.rmtree(TMP_DATASET_LOC + "_temp", ignore_errors=True)

    del sents


def tokenize(tokenizer, data_file_path, splits={"train": 0.8, "test": 0.2}):

    dataSet = datasets.load_from_disk(TMP_DATASET_LOC, keep_in_memory=False)

    def _tokenize(batch):
        encodings = tokenizer.encode_batch(batch)
        return {
            "input_ids": list(x.ids for x in encodings),
            "special_tokens_mask": list(x.special_tokens_mask for x in encodings),
            "input_ids_len": [len(x.ids) for x in encodings],
        }

    dataSet = dataSet.map(
        _tokenize,
        batched=True,
        batch_size=100000,
        input_columns=[TAG],
        remove_columns=[TAG],
    )

    dataSet = dataSet.shard(num_shards=4, index=0)
    splits = dataSet.train_test_split(
        train_size=splits["train"], test_size=splits["test"], seed=42
    )

    splits.save_to_disk(f"{data_file_path}/dataset")


def tokenize_fast(
    tokenizer, data_file_path, max_seq_length, splits={"train": 0.8, "test": 0.2}
):

    dataSet = datasets.load_from_disk(TMP_DATASET_LOC, keep_in_memory=False)

    def _tokenize_fast(batch):
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
        return {
            "input_ids": encodings["input_ids"],
            "special_tokens_mask": encodings["special_tokens_mask"],
            "input_ids_len": [len(x) for x in encodings["input_ids"]],
        }

    dataSet = dataSet.map(
        _tokenize_fast,
        batched=True,
        batch_size=100000,
        input_columns=[TAG],
        # remove_columns=[TAG],
        num_proc=10,
    )
    print("Sharding..")
    dataSet = dataSet.shard(num_shards=4, index=0)
    splits = dataSet.train_test_split(
        train_size=splits["train"], test_size=splits["test"], seed=42
    )

    print("Saving..")
    splits.save_to_disk(f"{data_file_path}/dataset")

