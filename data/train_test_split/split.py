import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

train = train.collect()
test = test.collect()
valid = valid.collect()

def visualize():
    _, axs = plt.subplots(3, len(train.columns) - 1, figsize=(11.69,8.27))
    colors = ['gray', 'yellow', 'green', 'red']
    labels = ['NOT-PRESENT', 'NEUTRAL', 'POSITIVE', 'NEGATIVE']

    for i, (df, title) in enumerate(zip([train, test, valid], ['Train', 'Test', 'Valid'])):
        for j, column in enumerate(c for c in df.columns if c != 'text'):
            d = df[column].value_counts().rows_by_key(key=column) # type: ignore
            d[0] = d.pop(None)

            kv = [(int(k), list(v)[0]) for k, v in d.items()]

            kv.sort(key=lambda x: x[0])
            _, values = zip(*kv)
            
            axs[i, j].pie(
                values, labels=None,
                colors=colors
            )
            if i == 0:
                axs[i, j].set_title(
                    column.replace('__', '\n'), 
                    fontsize=6, 
                    fontweight='bold'
                )

        legend_patches = [
            mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)
        ]
        plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.3, 0))

        axs[i, 0].text(
            -0.5, 0.5, title, 
            rotation=90, va='center', ha='center', 
            fontsize=12, fontweight='bold', 
            transform=axs[i, 0].transAxes
        )

    plt.subplots_adjust(
        hspace=0.0, wspace=0.0, 
        top=1.0, bottom=0.0,
        left=0.05, right=0.95
    )
    plt.tight_layout()
    plt.show()

def visualize_gray():
    _, axs = plt.subplots(3, len(train.columns) - 1, figsize=(11.69,8.27))
    hatches = ['', '||||', 'xxxx', '....']
    gray_colors = ['lightgray', 'darkgray', 'dimgray', 'gray']
    labels = ['NOT-PRESENT', 'NEUTRAL', 'POSITIVE', 'NEGATIVE']

    for i, (df, title) in enumerate(zip([train, test, valid], ['Train', 'Test', 'Valid'])):
        for j, column in enumerate(c for c in df.columns if c != 'text'):
            d = df[column].value_counts().rows_by_key(key=column) # type: ignore
            d[0] = d.pop(None)

            kv = [(int(k), list(v)[0]) for k, v in d.items()]

            kv.sort(key=lambda x: x[0])
            _, values = zip(*kv)
           
            wedges, texts = axs[i, j].pie(
                values, labels=None, colors=gray_colors
            )

            for wedge, hatch in zip(wedges, hatches):
                wedge.set_hatch(hatch)
            
            if i == 0:
                axs[i, j].set_title(
                    column.replace('__', '\n'), 
                    fontsize=6, 
                    fontweight='bold'
                )
        
        legend_patches = [
            mpatches.Patch(facecolor=color, hatch=hatch, label=label) for color, hatch, label in zip(gray_colors, hatches, labels)
        ]
        plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.3, 0))

        axs[i, 0].text(
            -0.5, 0.5, title, 
            rotation=90, va='center', ha='center', 
            fontsize=12, fontweight='bold', 
            transform=axs[i, 0].transAxes
        )

    plt.subplots_adjust(
        hspace=0.0, wspace=0.0, 
        top=1.0, bottom=0.0,
        left=0.05, right=0.95
    )
    plt.tight_layout()
    plt.show()


visualize_gray()

train.write_csv('train.csv')
test.write_csv('test.csv')
valid.write_csv('valid.csv')
