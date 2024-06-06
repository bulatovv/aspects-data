from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import datasets
import numpy as np
import torch

BASE_MODEL = "seninoseno/rubert-base-cased-sentiment-study-feedbacks-solyanka"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, model_max_length=512)

dataset = datasets.load_dataset(
    "csv", 
    data_files={
        "train": "train_pairs.csv", 
        "test": "test_pairs.csv",
        "valid": "valid_pairs.csv"
    },
)


def preprocess_function(examples):
    return tokenizer(text=examples['text'], text_pair=examples['aspect'], truncation='only_first')


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

import matplotlib.pyplot as plt

# Функция для извлечения значений метрик из логов тренера
def extract_metric_from_log(log_history, metric_name):
    return [entry[metric_name] for entry in log_history if metric_name in entry]

# Извлекаем значения метрик
train_loss = extract_metric_from_log(trainer.state.log_history, 'loss')
eval_loss = extract_metric_from_log(trainer.state.log_history, 'eval_loss')
eval_f1 = extract_metric_from_log(trainer.state.log_history, 'eval_f1')

# Построение графиков
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 4))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, eval_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# График F1-метрики
plt.subplot(1, 2, 2)
plt.plot(epochs, eval_f1, label='Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Validation F1 Score')
plt.legend()

plt.tight_layout()

# Сохраняем график в файл
plt.savefig('training_history.png')
