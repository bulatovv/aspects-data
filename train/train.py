from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
from datasets import Dataset, ClassLabel
import polars as pl
import numpy as np

BASE_MODEL = "seninoseno/rubert-base-cased-sentiment-study-feedbacks-solyanka"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, model_max_length=512)


train = Dataset(pl.read_csv('train_pairs.csv').to_arrow())
valid = Dataset(pl.read_csv('valid_pairs.csv').to_arrow())

train.cast_column('sentiment', ClassLabel(num_classes=2))
valid.cast_column('sentiment', ClassLabel(num_classes=2))


def preprocess_function(examples):
    return tokenizer(examples['text'], examples['aspect'], truncation='only_first')


tokenized_train = train.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "NOT-PRESENT", 1: "NEUTRAL", 2: "POSITIVE", 3: "NEGATIVE"}
label2id = {"NOT-PRESENT": 0, "NEUTRAL": 1, "POSITIVE": 2, "NEGATIVE": 3}

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, 
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision_score = precision.compute(predictions=predictions, references=labels)
    recall_score = recall.compute(predictions=predictions, references=labels)
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
    train_dataset=train,
    eval_dataset=valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)

trainer.train()
