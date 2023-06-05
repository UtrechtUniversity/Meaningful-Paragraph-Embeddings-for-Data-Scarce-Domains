# Meaningful-Paragraph-Embeddings-for-Data-Scarce-Domains
Code for the paper *Meaningful Paragraph Embeddings for Data Scarce Domains* for the 6th Workshop on Automated Semantic Analysis of Information in Legal Text.

## Scripts
All relevant python scripts for this project.
The BERT folder contains all scripts for training BERT models, and embedding text data using these models.
- convert_to_long: script for converting an existing BERT model to a LongFormer model.
- extend_tokenizer.py: script for extending an existing tokenizer.
- tokenizer_stats: script for analysing tokenizer performance.
- pre_process.py: script used to preprocess data for use in the pre_training.py script.
- pre_training.py: script for (further) pretraining an BERT model.

Other scripts:
- data.py: script for embedding a dataset using a BERT model, for use in the classification_model.py script. Note that embedding vectors are created only once.
- classification_model.py: the script for training a classification model using some set of embeddings.
- model_performance.py: script used for evaluating a trained classification model.

## Data
- ecthr_cases_violated: the dataset used in experiments. You can access the dataset using
  ```
  import datasets
  data = datasets.load_from_disk("/data/ecthr_cases_violated")
  ```
 - ecthr_cases_train_sents.txt: the dataset used to furter pretrain our BERT models.

## Results
  .csv files containing classification results per classification model. Furthermore, the state dicts of models are stored in this folder.
  You can load a model using
  ```
  import torch
  from classification_model import BasicModel
  model = BasicModel(10)
  model.load_state_dict(torch.load(PATH))
  ```
  Every classification model expects a (padded) ndarray of embeddings.
  See classification_model.py for more details.

Feel free to use our code, but please use the following citation: 

```
@inproceedings{herrewijnen2023meaningfulDomainEmbeddings,
  title="Towards Meaningful Paragraph Embeddings for Data-Scarce Domains: a Case Study in the Legal Domain",
  author={Herrewijnen, Elize and Craandijk, Dennis F W},
  booktitle={6th Workshop on Automated Semantic Analysis of Information in Legal Text},
  maintitle = {International Conference on Artificial Intelligence and Law 2023 (ICAIL 2023)},
  year={2023},
  organization={ICAIL}
}
```
