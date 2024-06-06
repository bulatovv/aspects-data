import polars as pl
import numpy as np
import torch
from ctgan import CTGAN
from functools import reduce
from operator import mul

SEED_VALUE = 42

np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

q = pl.scan_csv('data.csv').select(pl.exclude('text')).with_columns(pl.col('эссе').cast(pl.Int64).alias('эссе'))

freq = q.melt().drop_nulls().group_by('variable', 'value').len()
freq = freq.with_columns(score=(pl.sum('len') / pl.col('len')).log()).select(pl.concat_str('variable', 'value'), 'score')
freq = freq.collect().rows_by_key('variable', unique=True)

def generate_score_expr(null_score):
    return reduce(mul, [
        pl.when(
            pl.col(c) == 1
        ).then(
            pl.lit(freq[c + '1'])
        ).when( 
            pl.col(c) == 2
        ).then(
            pl.lit(freq[c + '2'])
        ).when( 
            pl.col(c) == 3
        ).then(
            pl.lit(freq[c + '3'])
        ).otherwise(pl.lit(null_score).log())
        for c in q.columns
    ])

ctgan = CTGAN(epochs=10)

real_data = q.collect()
ctgan.fit(real_data.to_pandas(), discrete_columns=real_data.columns)

synthetic_data = pl.from_pandas(ctgan.sample(10000)).lazy()

synthetic_data = synthetic_data.fill_nan(0).filter(
    pl.all_horizontal(pl.all().is_nan()).not_()
).with_columns(
    score=generate_score_expr(null_score=300)
).sort('score', descending=True).limit(1000).select(pl.exclude('score'))

synthetic_data.sink_csv('synt.csv')
