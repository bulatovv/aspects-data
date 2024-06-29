from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from functools import cache
import evaluate
import datasets
import numpy as np
import torch

BASE_MODEL = "DeepPavlov/rubert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, model_max_length=512)

dataset = datasets.load_dataset(
    "csv", 
    data_files={
        "train": "train_pairs.csv", 
        "test": "test_pairs.csv",
        "valid": "valid_pairs.csv"
    },
)

@cache
def preprocess_label(label):
    return label.replace('__', ', ')


def preprocess_function(examples):
    return tokenizer(
        text=examples['text'],
        text_pair=list(map(preprocess_label, examples['aspect'])),
        truncation='only_first'
    )


tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "NOT-PRESENT", 1: "NEUTRAL", 2: "POSITIVE", 3: "NEGATIVE"}
label2id = {"NOT-PRESENT": 0, "NEUTRAL": 1, "POSITIVE": 2, "NEGATIVE": 3}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, 
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

model.to(device)

precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision_score = precision.compute(predictions=predictions, references=labels, average='macro')
    recall_score = recall.compute(predictions=predictions, references=labels, average='macro')
    f1_score = f1.compute(predictions=predictions, references=labels, average='macro')

    return {'precision' : precision_score, 'recall' : recall_score, 'f1' : f1_score}

training_args = TrainingArguments(
    output_dir="study_absa",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'], # type ignore
    eval_dataset=tokenized_dataset['valid'], # type ingore
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)

trainer.train()


import pandas as pd

# Извлечение истории обучения
log_history = trainer.state.log_history

# Преобразование истории в DataFrame
df = pd.DataFrame(log_history)

# Сохранение DataFrame в CSV файл
df.to_csv('training_history.csv', index=False)
