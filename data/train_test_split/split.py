import polars as pl
import numpy as np
import matplotlib.pyplot as plt


def stratified_split(
    q: pl.LazyFrame, 
    ratio: list[float], 
    x_cols: list[str],
    seed: int|None = None
) -> list[pl.LazyFrame]:
    gen = np.random.default_rng(seed)
    

    q = q.group_by(pl.exclude(x_cols), maintain_order=True).map_groups(
        lambda df: df.with_columns(
            split_label=pl.lit(gen.choice(
                list(range(len(ratio))),
                size=df.height,
                p=ratio
            ))
        ),
        schema=q.schema | {'split_label': pl.Int64}
    )

    return [
        q
        .filter(
            pl.col('split_label') == i
        ).select(pl.exclude('split_label'))
        .sort('text', maintain_order=True)
        for i in range(len(ratio))
    ]





q = pl.scan_csv('data.csv')

train, test, valid = stratified_split(
    q, [0.8, 0.1, 0.1], ['text'], seed=42
)

fig, axs = plt.subplots(ncols=len(train.columns), nrows=3, figsize=(190, 30),
                        layout="constrained")


colors = ['gray', 'yellow', 'green', 'red']


train = train.collect()
test = test.collect()
valid = valid.collect()

def visualize():
    for i, df in enumerate([train, test, valid]):
        for j, column in enumerate(df.columns):
            if column != 'text':
                d = df[column].value_counts().rows_by_key(key=column)
                d[0] = d.pop(None)
                
                
                kv = [(int(k), list(v)[0]) for k, v in d.items()]
                kv.sort(key=lambda x: x[0])
                keys, values = zip(*kv)

                axs[i, j].pie(
                    values, labels=keys,
                    colors=colors
                )

    plt.show()


train.write_csv('train.csv')
test.write_csv('test.csv')
valid.write_csv('valid.csv')
