import torch
from transformers import BertTokenizer, BertForSequenceClassification
import polars as pl
import polars.selectors as cs

from torch.utils.data import DataLoader, TensorDataset
# Load model and tokenizer
model_name = "DeepPavlov/rubert-base-cased"
checkpoint_dir = "./checkpoint-3960"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load test data
q = pl.scan_csv('test.csv').fill_null(0)

# Prepare text-aspect pairs
# aspects = ['лекции', 'доклады', 'проекты', 'презентации', 'фильмы', 'видео-уроки', 'задания__задачи', 'онлайн-курс', 'баллы__оценки', 'практики__семинары', 'тесты', 'домашняя работа', 'эссе', 'выступления', 'зачет__экзамен', 'материал__информация__темы', 'литература__учебники', 'игры__интерактивность', 'преподаватель']

preproc = q.melt('text', variable_name='aspect', value_name='sentiment').collect()

inputs = tokenizer(list(preproc['text']), list(preproc['aspect']), 
                   padding=True, truncation=True, return_tensors="pt", max_length=512)

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Predict
predictions = []
with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask, token_type_ids = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)
        predictions.extend(pred.cpu().tolist())


preproc = preproc.with_columns(
    pred=pl.Series(predictions)
)

preproc.write_csv('predicts.csv')
