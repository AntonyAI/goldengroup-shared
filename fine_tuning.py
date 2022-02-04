import datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
)

import pandas as pd
import wandb
import nltk
import numpy as np
import torch

torch.cuda.empty_cache()

from datasets import load_metric, load_dataset
from datasets.arrow_dataset import Dataset

# rouge loss function
metric = load_metric("rouge")


# inizializza la sessione wandb
wandb.init(project="my-test-project", entity="antonyai")

# nome del modello
checkpoint = "t5-base"

# t5 seq2seq model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# il mio dataset
dati_train = load_dataset("json", data_files="dialogsum.train.jsonl")

dati_val = load_dataset("json", data_files="dialogsum.dev.jsonl")

print(dati_train)


def preprocess_function(examples):
    inputs = ["summarize:" + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# otteniamo i dataset dalle funzioni appena scritte

tokenized_train = dati_train.map(preprocess_function, batched=True)
tokenized_val = dati_val.map(preprocess_function, batched=True)

print(tokenized_val["train"]["summary"])


batch_size_train = 16
batch_size_val = 6
model_name = "t5_base_tuned"
args = Seq2SeqTrainingArguments(
    model_name,
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size_train,
    per_device_eval_batch_size=batch_size_val,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
    report_to="wandb",
)

# metrics: rouge loss
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train["train"],
    eval_dataset=tokenized_val["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
