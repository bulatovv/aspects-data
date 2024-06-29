import evaluate
import polars as pl

q = pl.scan_csv('predicts.csv')

f1 = evaluate.load("f1")

# class presense f1
presense = q.with_columns(
    pred=pl.col('pred').eq(0).not_().cast(pl.Int64),
    sentiment=pl.col('sentiment').eq(0).not_().cast(pl.Int64),
).collect()

presense_score = f1.compute(
    references=presense['sentiment'], 
    predictions=presense['pred']
)
print(presense_score)


# f1 for found
found = q.filter(pl.col('pred') != 0).collect()
found_score = f1.compute(
    references=found['sentiment'], 
    predictions=found['pred'], 
    average='macro'
)
print(found_score)


# per sentiment score
for i in range(4):
    #print(sent.collect())
    sent = q.with_columns(
        sentiment=pl.col('sentiment').eq(i).cast(pl.Int64),
        pred=pl.col('pred').eq(i).cast(pl.Int64)
    ).collect()
    #print(sent)
    sent_score = f1.compute(
        references=sent['sentiment'], 
        predictions=sent['pred'], 
    )
    print(f"sentiment {i}:", sent_score)



df = q.collect()
score = f1.compute(references=df['sentiment'], predictions=df['pred'], average='macro')

print(score)
