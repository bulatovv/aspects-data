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

# Augmentation
AUGMENTATION_OUT_PATH = Path(DATA_DIRECTORY) / "out_augmented.csv"

# Generation
GENERATION_OUT_PATH = Path(DATA_DIRECTORY) / "out_generated.csv"

# OpenAI
OPENAI_REQUEST_DELAY = 0
OPENAI_KEY = env["OPENAI_KEY"]
OPENAI_URL = env["OPENAI_URL"]
OPENAI_MODEL_NAME = env["OPENAI_MODEL_NAME"]
