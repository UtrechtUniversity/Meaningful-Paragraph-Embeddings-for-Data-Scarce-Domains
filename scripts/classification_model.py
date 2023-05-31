import collections
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List
from ml_utils.model_training import torch_trainer
import torch.utils.data
import datasets
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import default_collate
import pandas as pd
from sklearn.metrics import classification_report, f1_score


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        vectors: np.array,
        labels: np.array,
        sentences: Iterable[List[str]],
        device: str,
    ):
        "Initialization"
        self.labels = labels
        self.device = device
        self.vectors = vectors
        self.num_sentences = [len(x) for x in sentences]

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "inputs": np.vstack(self.vectors[index]),
            "labels": self.labels[index],
            "train_labels": torch.from_numpy(self.labels[index]).float(),
            "indices": index,
            "num_sentences": self.num_sentences[index],
        }


class BasicModel(torch_trainer.BaseModule):
    def __init__(self, dims):
        super(BasicModel, self).__init__()
        self.dims = dims
        self.flat = nn.Flatten()
        self.conv1 = nn.Conv1d(768, 768, kernel_size=(1))
        self.conv2 = nn.Conv1d(768, 768, kernel_size=(1))
        self.conv3 = nn.Conv1d(768, 768, kernel_size=(1))
        self.lin1 = nn.Linear(int(768 / 1), int(768 / 1))
        self.lin2 = nn.Linear(int(768 / 1), int(768 / 1))
        self.lin = nn.Linear(int(768 / 1), dims)
        self.metric = 0

    def forward(self, x):
        if x.device != self.device:
            x.to(self.device)
        x = x.float()
        x = torch.transpose(x, 1, 2)
        F.leaky_relu(self.conv1(x), inplace=True)
        F.leaky_relu(self.conv2(x), inplace=True)
        F.leaky_relu(self.conv3(x), inplace=True)
        x = torch.transpose(x, 1, 2)
        x = self.lin2(x)
        x = self.lin1(x)
        x = self.lin(x)
        return x

    def get_dims(self):
        return self.dims

    @property
    def name(self):
        return "BasicModel"

    def activation(self, inputs, mean=True):
        if mean:
            predictions = inputs.mean(1).float()
        else:
            predictions = inputs
        return torch.sigmoid(predictions)

    def predict(self, input, explain=False, **kwargs):
        logits = self(input)
        activated = self.activation(logits, not explain).cpu().detach()
        return activated.round()


def collate_fn(batch):
    first_elem = batch[0]
    elem_type = type(first_elem)
    if isinstance(first_elem, torch.Tensor) and any(
        first_elem.shape != elem_.shape for elem_ in batch
    ):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

    if isinstance(first_elem, np.ndarray) and any(
        first_elem.shape != elem_.shape for elem_ in batch
    ):
        batch = [torch.from_numpy(b) for b in batch]
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

    elif isinstance(first_elem, collections.abc.Mapping):
        try:
            return elem_type(
                {key: collate_fn([d[key] for d in batch]) for key in first_elem}
            )
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate_fn([d[key] for d in batch]) for key in first_elem}
    else:
        return default_collate(batch)


def get_class_weights(class_series):
    n_samples = len(class_series)

    if hasattr(class_series[0], "__len__"):
        n_classes = len(class_series[0])

        # Count each class frequency
        class_count = [0] * n_classes
        for classes in class_series:
            for index in range(n_classes):
                if classes[index] != 0:
                    class_count[index] += 1
                    # Compute class weights using balanced method
        class_weights = [
            n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count
        ]
    else:
        n_classes = len(set(class_series))
        from sklearn.utils import class_weight

        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(class_series), y=class_series
        )

    class_labels = range(len(class_weights))
    return dict(zip(class_labels, class_weights))


def make_binary(example):
    x = int(any(example["violated_articles"]))
    example["binary_labels"] = np.array([x])
    return example


def filter_occ(example):
    x = set(example["violated_articles"])
    filter_ = set(["11", "14", "10", "2", "8", "13", "3", "5", "P1-1", "6"])
    example["filtered_violated_articles"] = x - (x - filter_)
    return example


if __name__ == "__main__":

    encoders = [
        "bert_base_uncased",
        # "legal_bert_base_uncased",
        # "ecthr_base_ex",
        # "roberta_base",
        # "longformer_base",
        # "ecthr_robert_long",
    ]
    for encoder in encoders:
        data = datasets.load_from_disk(
            "data/ecthr_cases_violated_" + encoder
        )

        data = data.map(filter_occ)
        label = "filtered_violated_articles"

        label_binarizer = MultiLabelBinarizer()
        label_binarizer = label_binarizer.fit(data["train"][label])

        trainDataset = Dataset(
            data["train"]["vectors"],
            label_binarizer.transform(data["train"][label]).astype(float),
            data["train"]["gold_rationales"],
            "cuda",
        )

        testDataset = Dataset(
            data["test"]["vectors"],
            label_binarizer.transform(data["test"][label]).astype(float),
            data["test"]["gold_rationales"],
            "cuda",
        )

        valDataset = Dataset(
            data["dev"]["vectors"],
            label_binarizer.transform(data["dev"][label]).astype(float),
            data["dev"]["gold_rationales"],
            "cuda",
        )

        class_weights = get_class_weights(trainDataset.labels)
        model = BasicModel(len(class_weights)).cuda()
        criterion = nn.BCELoss(weight=torch.tensor(list(class_weights.values())).cuda())

        optimizer = optim.Adam(model.parameters())

        BATCH_SIZE = 64
        testdataLoader = torch.utils.data.DataLoader(
            testDataset, collate_fn=collate_fn, shuffle=False, batch_size=BATCH_SIZE
        )
        valdataLoader = torch.utils.data.DataLoader(
            valDataset, collate_fn=collate_fn, shuffle=False, batch_size=BATCH_SIZE
        )
        traindataLoader = torch.utils.data.DataLoader(
            trainDataset, shuffle=True, collate_fn=collate_fn, batch_size=BATCH_SIZE
        )

        torch_trainer.train_model(
            model,
            traindataLoader=traindataLoader,
            testdataLoader=valdataLoader,
            criterion=criterion,
            optimizer=optimizer,
            activation=model.activation,
            device="cuda",
            epochs=15,
        )

        print(encoder)
        test_res = torch_trainer.validate_model(
            model, testdataLoader, criterion, model.activation, "cuda"
        )

        clsf_report = pd.DataFrame(
            classification_report(
                y_true=test_res[1], y_pred=test_res[0], output_dict=True
            )
        ).transpose()
        clsf_report.index = list(label_binarizer.classes_) + list(
            clsf_report.index[-4:]
        )

        clsf_report["tes_res"] = pd.Series([test_res[0]], index=clsf_report.index[[0]])
        clsf_report["tes_label"] = pd.Series(
            [test_res[1]], index=clsf_report.index[[0]]
        )
        clsf_report.to_csv(encoder + ".csv", index=True)
        torch.save(model.state_dict(), encoder + "_classification_model.pt")
        print(clsf_report["f1-score"])

