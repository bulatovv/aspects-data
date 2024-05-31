import dotenv
from pathlib import Path

env = dotenv.dotenv_values(".env")

# External data
DATA_DIRECTORY = "data"
DATA_ANNOTATED_URL = env["DATA_ANNOTATED_URL"]
DATA_ANNOTATED_FILENAME = "annotated.csv"
DATA_ELECTIVE_NAMES_URL = env["DATA_ELECTIVE_NAMES_URL"]
DATA_ELECTIVE_NAMES_FILENAME = "elective_names.csv"
DATA_ANNOTATED_GENERATION_URL = env["DATA_ANNOTATED_GENERATION_URL"]
DATA_ANNOTATED_GENERATION_FILENAME = "annotated_generation.csv"

# Prompts
PROMPT_AUGMENTATION_SYSTEM_PATH = "prompts/prompt_augmentation_system.txt"
PROMPT_AUGMENTATION_CLIENT_PATH = "prompts/prompt_augmentation_client.txt"
PROMPT_GENERATION_SYSTEM_PATH = "prompts/prompt_generation_system.txt"
PROMPT_GENERATION_CLIENT_PATH = "prompts/prompt_generation_client.txt"
PROMPT_GENERATION_FEW_SHOT_CLIENT_1 = \
"""
! Психика и мозг

Задания и задачи: нейтральный
Быллы и оценки: негативный
Практики и семинары: позитивный
Зачёт и экзамен: негативный
Материал и информация и темы: позитивный
"""
PROMPT_GENERATION_FEW_SHOT_ASSISTENT_1 = \
"""
Без биологической школьной базы вам будет тяжело на этом элективе. Автомат на 3 гарантирован, если ходите на все пары и сдаете все практические работы по всем правилам. Если хотите закрыться на 4 или 5, то придется выходить на зачет. Баллы зажимает и выставляет все спустя месяц-два, что очень пошатывает нервы и демотивирует делать задания. Лекции Плотникова довольно монотонно рассказывает, но информация познавательная. На практических занятиях достаточно интересно: вы будете делать задания по исследованию своего внимания, памяти, сенсорных систем, интеллекта, воле, эмоциям. Самым запоминающимся заданием для меня было выявления нарушений мышления (мы выявляли по видеоматериалам нарушения мышления у людей с тотальным слабоумием, шизофренией, олигофренией, эпилепсией)
"""
PROMPT_GENERATION_FEW_SHOT_CLIENT_2 = \
"""
! Компьютерная математика

Задания и задачи: негативный
Баллы и оценки: негативный
Домашняя работа: нейтральный
Преподаватель: негативный
"""
PROMPT_GENERATION_FEW_SHOT_ASSISTENT_2 = \
"""
Самый худший электив в моей жизни.Отобрал у меня последние нервные клетки и время.Каждый понедельник был днём слез,потому что осилить уровень математики якобы «для гуманитариев» я была не в силах,при том,что не плохо соображала. У разных преподавателей были разные требования.Я попала к тому,у кого они завышены. Очень много задавалось на дом,невероятно сложные порой задания,было такое,что за 4 выполненых задания с ошибкой где-то в расчётах в начале,я получила 0,5))))Возможности получить доп.баллы не предоставлялось,также как и перерешать задания.Отношение было наплевательское,темы никто не разжевывал,все приходилось разбирать дома самой; чрезмерный контроль на контрольных работах.
"""
PROMPT_GENERATION_FEW_SHOT_CLIENT_3 = \
"""
! Психология общения

Лекции: нейтрально
Фильмы: позитивно
практики и семинары: позитивно
Тесты: позитивно
Преподаватель: нейтрально
"""
PROMPT_GENERATION_FEW_SHOT_ASSISTENT_3 = \
"""
Лекции ходила к одному преподавателю, а практики к другому.
Практика вообще чудо!!! Тестики всякие разные очень интересно и не сильно энергозатратно.
Задали как-то посмотреть первую серию сериала, вообще крутой сериал, посмотрела все сезоны. Но мне кажется зависит от преподавателя.
"""

# Augmentation
AUGMENTATION_OUT_PATH = Path(DATA_DIRECTORY) / "out_augmented.csv"

# Generation
GENERATION_OUT_PATH = Path(DATA_DIRECTORY) / "out_generated.csv"

# OpenAI
OPENAI_REQUEST_DELAY = 0
OPENAI_KEY = env["OPENAI_KEY"]
OPENAI_URL = env["OPENAI_URL"]
OPENAI_MODEL_NAME = env["OPENAI_MODEL_NAME"]
