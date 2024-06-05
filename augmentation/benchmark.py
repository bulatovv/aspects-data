from natasha import (
    Segmenter,
    MorphVocab, 
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
import numpy as np

import bert_score
def custom_bertscore_tokenizer(tokenizer, sent):
    return tokenizer.encode(
        sent,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
    )

# monkey patch bertscore tokenizer
bert_score.sent_encode.__code__ = custom_bertscore_tokenizer.__code__


from transformers import AutoTokenizer
from evaluate import load

bertscore_metric = load('bertscore')
bleu_metric = load('bleu')

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)






def tokenize(text: str) -> list[str]:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens: # type: ignore
        token.lemmatize(morph_vocab)

    return [
        token.lemma for token in doc.tokens # type:ignore
        if token.pos not in ['PUNCT', 'ADP', 'CCONJ']
    ]

def paraphrase_metrics(
    references: list[str], 
    candidates: list[str]
) -> dict[str, float]:

    bertscore = bertscore_metric.compute(
        predictions=candidates, 
        references=references,
        model_type='DeepPavlov/rubert-base-cased',
        num_layers=9,
        use_fast_tokenizer=True
    )['f1']
    bleu = bleu_metric.compute(
        predictions=candidates,
        references=[[r] for r in references],
        tokenizer=tokenize
    )['bleu']

    a = 1 - np.array(bleu)
    b = np.array(bertscore)
    combined = 2 * a * b / (a + b)

    return {
        'berstcore': np.average(bertscore),
        'bleu': np.average(bleu),
        'combined': np.average(combined) # type: ignore
    }
