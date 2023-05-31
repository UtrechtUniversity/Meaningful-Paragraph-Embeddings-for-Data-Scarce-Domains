import json
import os
import datasets
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    BertForPreTraining,
)
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

data_name = "bert_ecthr_ex_sents"
tag = "facts"
model_name = "bert-base-multilingual-cased"
tokenizer_name = model_name
max_seq_length = 512

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=512)
model = BertForPreTraining.from_pretrained(model_name)
output_dir = "results"
train_data = ["../data/ecthr_cases_train_sents.txt"]
data_file_path = "../data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(data_file_path):
    from pre_process import pre_process_data

    pre_process_data(
        data_file_path,
        train_data,
        tag,
        tokenizer,
        output_dir,
        max_seq_length=max_seq_length,
    )


def tokenizer_init():
    return AutoTokenizer.from_pretrained(
        tokenizer_name, model_max_length=max_seq_length
    )


def model_init():
    model = BertForPreTraining.from_pretrained(model_name, return_dict=True)
    model.resize_token_embeddings(len(tokenizer_init()))
    return model


dataSet = datasets.load_from_disk("/dataset")

cols_to_remove = set(dataSet["train"].features.keys()) - {
    "input_ids",
    "next_sentence_label",
    "special_tokens_mask",
}

dataSet = dataSet.remove_columns(cols_to_remove)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
batch_size = 16


def compute_metrics(p):
    print(p)
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {
        "eval_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    prediction_loss_only=True,
    do_train=True,
    save_total_limit=5,
    logging_dir="./logs",
    remove_unused_columns=True,
    fp16=True,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    eval_accumulation_steps=16,
)


trainer = Trainer(
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataSet["train"],
    eval_dataset=dataSet["test"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)

try:
    trainer.train()

finally:
    eval_loss = trainer.evaluate()
    trainer.save_state()
    trainer.save_model(output_dir + "/final")
    tokenizer.save_pretrained(output_dir + "/final")
    with open(output_dir + "/model_eval.json", "w") as f:
        json.dump(trainer.save_metrics("all"), f)

