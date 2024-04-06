from os import listdir
from os.path import isfile, join
from pathlib import Path
import re

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd

from tqdm import tqdm

# Подгружаем все размеченные аспекты из файлов и сохраняем в памяти

DATA_DIRECTORY = "research/stage_2_keywords_search/search_annotated/"

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Предварительно удаляем расщирение .txt
aspect_file_names = [
    f[:-4]
        for f
        in listdir(DATA_DIRECTORY)
        if isfile(join(DATA_DIRECTORY, f))
]

# Объект аспект - массив предложений
actual_aspects_sentences = {}
# Список всех уникальных предложений
all_aspects_senteces = []
for file_name in aspect_file_names:
    with open(DATA_DIRECTORY+file_name+".txt", "r", encoding="utf-8") as f:
        actual_aspects_sentences[file_name] = set(f.read().split('\n'))
        all_aspects_senteces += actual_aspects_sentences[file_name]

all_aspects_senteces = list(set(all_aspects_senteces))
actual_aspects_sentences.pop("мусор")


def load_aspects():
    return list(actual_aspects_sentences.keys())

def clean_text(text: str):
    # Убираем ссылки
    clean = re.sub(u'(http|ftp|https):\/\/[^ ]+', '', text)
    # Убираем все неалфавитные символы кроме дефиса и апострофа
    clean = re.sub(u'[^а-я^А-Я^\w^\-^\']', ' ', clean)
    # Убираем тире
    clean = re.sub(u' - ', ' ', clean)
    # Убираем дубликаты пробелов
    clean = re.sub(u'\s+', ' ', clean)
    # Убираем пробелы в начале и в конце строки
    clean = clean.strip().lower()
    return clean

# TODO сделать поддержку синонимов

class MethodSubstring():
    dictionary = {
        "материал информация": ["материал", "информация", "теория"],
        "домашняя работа": ["домашняя работа", "домашнее задание", "дз", "домашка", "д/з"],
        "зачёт": ["зачет", "зачёт", "экзамен", "комиссия", "закрыться"],
        "фильмы": ["фильм", "кино"],
        "презентации": ["презентация", "преза"],
        "онлайн-курс": ["онлайн-курс", "дистанционно", "онлайн"],
        "видео-уроки": ["видео-урок", "видеоурок", "видео", "видео-лекция", "видеолекция"],
        "преподаватель": ["преподаватель", "препод"],
        "выступления": ["выступление", "выступать"],
        "литература": ["литература", "книга"],
        "тесты": ["тест", "тестирование"],
        "практики семинары": ["практика", "семинар", "практическая"],
        "доклады": ["доклад"],
        "задания задачи": ["задание", "задача"],
        "баллы": ["балл", "оценка"],
        "эссе": ["эссе", "сочинение"],
        "проекты": ["проект", "проектная работа"],
        "игры" : ["игра"],
        "лекции": ["лекция"],
    }


    def set_aspects(self, aspects_list=None):
        if aspects_list is not None:
            self.aspects_list = aspects_list
        else:
            self.aspects_list = load_aspects()

    def __init__(self):
        self.set_aspects(None)

    def find_aspects(self, sentence: str):
        aspects = []
        # Очистка
        sentence_words = clean_text(sentence).split(" ")
        # Лемматизация
        doc = Doc(sentence)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        for token in doc.tokens: 
            token.lemmatize(morph_vocab)
        
        lemmatized = ' '.join(token.lemma for token in doc.tokens)
        
        for aspect in self.aspects_list:
            for word in self.dictionary[aspect]:
                if word in lemmatized:
                    aspects.append(aspect)
        return aspects

    def process(self, text: str):
        return self.find_aspects(text)

class MethodSimilarity():

    def transformers_tokenizer(self, sentence: str) -> torch.Tensor:
        tokens = self.transformers_auto_tokenizer(sentence, return_tensors='pt')
        vector = self.transformers_model(**tokens)[0].detach().squeeze()
        return torch.mean(vector, dim=0).numpy()

    def set_aspects(self, aspects_list=None):
        if aspects_list is not None:
            self.aspects_list = aspects_list
        else:
            self.aspects_list = load_aspects()

    def __init__(self, tokenizer="distiluse"):
        self.set_aspects(None)

        self.aspects_list = load_aspects()

        if tokenizer == "distiluse":
            self.tokenizer = self.transformers_tokenizer
            self.transformers_auto_tokenizer = AutoTokenizer.from_pretrained("ai_models/sentence-transformers/distiluse-base-multilingual-cased-v2", local_files_only=True)
            self.transformers_model = AutoModel.from_pretrained("ai_models/sentence-transformers/distiluse-base-multilingual-cased-v2", local_files_only=True)
        elif tokenizer == "sbert-pq":
            self.tokenizer = self.transformers_tokenizer
            self.transformers_auto_tokenizer = AutoTokenizer.from_pretrained("ai_models/inkoziev/sbert_pq", local_files_only=True)
            self.transformers_model = AutoModel.from_pretrained("ai_models/inkoziev/sbert_pq", local_files_only=True)
        else:
            self.tokenizer = None
            raise Exception("Invalid tokenizer.")

    def calc_similarity(self, vector1, vector2):
        return np.dot(vector1, vector2) / \
               (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def find_aspects(self, sentence: str, min_similarity: int):
        aspects = []
        # Схожесть
        sentence_vector = self.tokenizer(sentence)
        similarities = [(aspect, self.calc_similarity(self.tokenizer(aspect), sentence_vector))
                          for aspect in self.aspects_list]
        for similarity in similarities:
            if similarity[1] > min_similarity:
                aspects.append(similarity[0])
        return aspects

    def process(self, text: str, min_similarity: int = 0.3):
        return self.find_aspects(text, min_similarity)


search_substring = MethodSubstring()
substring_acc_list = []

search_similarity = MethodSimilarity()
similarity_acc_list = []

print("Initialized")

def make_report(method):
    y_expected = []
    y_predicted = []

    for sentence in tqdm(all_aspects_senteces):
        y_expected.append(
            [
                aspect
                for aspect in actual_aspects_sentences
                if sentence in actual_aspects_sentences[aspect]
            ]
        )
        y_predicted.append(method.process(sentence))

    y_expected = MultiLabelBinarizer(classes=list(actual_aspects_sentences)).fit_transform(y_expected)
    y_predicted = MultiLabelBinarizer(classes=list(actual_aspects_sentences)).fit_transform(y_predicted)
        
    report = classification_report(y_expected, y_predicted, target_names=list(actual_aspects_sentences), output_dict=True)
    return report

def save_report(report, name):
    df = pd.DataFrame(report).transpose()
# set precision for all numeric values to 2
    df[['precision', 'recall', 'f1-score']] = df[['precision', 'recall', 'f1-score']].applymap(lambda x: round(x, 2) if isinstance(x, float) else x)

    df.to_csv("scripts/output/report_" + name + ".csv")

print("Calculating substring method accuracy...")
report_substring = make_report(search_substring)
save_report(report_substring, "substring")

print("Calculating similarity method accuracy...")
report_similarity = make_report(search_similarity)
save_report(report_similarity, "similarity")
