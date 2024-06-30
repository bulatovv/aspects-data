import evaluate
from numpy import average
import polars as pl

q = pl.scan_csv('predicts.csv')

f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
accuracy = evaluate.load("accuracy")


# per sentiment score
for i in range(4):
    subset = q.with_columns(
        sentiment=pl.col('sentiment').eq(i).cast(pl.Int64),
        pred=pl.col('pred').eq(i).cast(pl.Int64)
    ).collect()
    f1_score = f1.compute(references=subset['sentiment'], predictions=subset['pred'])
    precision_score = precision.compute(references=subset['sentiment'], predictions=subset['pred'])
    recall_score = recall.compute(references=subset['sentiment'], predictions=subset['pred'])
    accuracy_score = accuracy.compute(references=subset['sentiment'], predictions=subset['pred'])
    print(f"sentiment {i}: {f1_score | precision_score | recall_score | accuracy_score}")
    
df = q.collect()
f1_score = f1.compute(references=df['sentiment'], predictions=df['pred'], average='macro')
precision_score = precision.compute(references=df['sentiment'], predictions=df['pred'], average='macro')
recall_score = recall.compute(references=df['sentiment'], predictions=df['pred'], average='macro')
accuracy_score = accuracy.compute(references=df['sentiment'], predictions=df['pred'])


print(f"avg for sentiment: f1 = {f1_score | precision_score | recall_score | accuracy_score}")

f1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []
"""
# per aspect score
for aspect in list(df['aspect'].unique()):
    subset = q.filter(pl.col('aspect') == aspect).collect()
    f1_score = f1.compute(references=subset['sentiment'], predictions=subset['pred'], average='macro')
    precision_score = precision.compute(references=subset['sentiment'], predictions=subset['pred'], average='macro')
    recall_score = recall.compute(references=subset['sentiment'], predictions=subset['pred'], average='macro')
    accuracy_score = accuracy.compute(references=subset['sentiment'], predictions=subset['pred'])
  
    print(f"aspect {aspect}: {f1_score | precision_score | recall_score | accuracy_score}")

    # Добавление средних значений в соответствующие списки
    f1_scores.append(f1_score['f1'])
    precision_scores.append(precision_score['precision'])
    recall_scores.append(recall_score['recall'])
    accuracy_scores.append(accuracy_score['accuracy'])

# Расчет среднего значения для каждой метрики
avg_f1_score = average(f1_scores)
avg_precision_score = average(precision_scores)
avg_recall_score = average(recall_scores)
avg_accuracy_score = average(accuracy_scores)

# Вывод средних значений
print(f"avg for aspects: {avg_f1_score}, {avg_precision_score}, {avg_recall_score}, {avg_accuracy_score}")
"""
