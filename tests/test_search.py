
from pathlib import Path

import pandas as pd

from ai.search import MethodSimilarity as Search

df = pd.read_csv("research/stage_1_keywords_extraction/Keyphrase extraction/data/all_dataset.csv", sep='\t', encoding="utf-8")

df = df[df["rubrics"].str.contains('образование_отзывус')].dropna().astype("str").reset_index()

text = ". ".join(df['text'])

search = Search()

print("Initialized")

RESULT_DIRECTORY = "scripts/output/test_search/"

Path(RESULT_DIRECTORY).mkdir(parents=True, exist_ok=True)

all_aspects = search.process(text, min_similarity=0.3)

for aspect, sentences in all_aspects.items():
    with open(RESULT_DIRECTORY + aspect + '.txt', 'w+', encoding="utf-8") as f:
        f.write('\n'.join(sentences))
