import pandas as pd
import config
from data_provider import DataProvider
from utilities import read_file

PROMPT_SYSTEM = read_file(config.PROMPT_GENERATION_SYSTEM_PATH)
PROMPT_CLIENT = read_file(config.PROMPT_GENERATION_CLIENT_PATH)
OUT_DIR = config.GENERATION_OUT_PATH

def run_generation(delay: float):
    elective_names = pd.read_csv(DataProvider.get_path("elective_names"), sep=",")["title"]
    print("start of generation")
