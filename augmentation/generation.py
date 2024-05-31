import os
import time
import pandas as pd
import random
import config
import openai_client
from data_provider import DataProvider
from utilities import read_file

PROMPT_SYSTEM = read_file(config.PROMPT_GENERATION_SYSTEM_PATH)
PROMPT_CLIENT = read_file(config.PROMPT_GENERATION_CLIENT_PATH)
OUT_DIR = config.GENERATION_OUT_PATH

def make_generation_request(current_feedback: int,
                            generation_df: pd.DataFrame,
                            course_name: str) -> str | None:
    label_to_sentiment = {
        1: "нейтральный",
        2: "позитивный",
        3: "негативный",
    }
    sentiments = [ i.replace("__", ", ").capitalize() + ": " + label_to_sentiment[v] for i, v in generation_df.iloc[current_feedback].replace(0, pd.NA).dropna().items() ]
    response = openai_client.gpt_request(
        system_prompt=PROMPT_SYSTEM,
        client_prompt=PROMPT_CLIENT.format(
            course_name = course_name,
            sentiment = "\n".join(sentiments)
        ),
        assistant_prompt="Отзыв студента на курс:\n",
    )
    return response

def run_generation(delay: float):

    random.seed(42)

    elective_names = pd.read_csv(DataProvider.get_path("elective_names"), sep=",")["title"]
    generation_df = pd.read_csv(DataProvider.get_path("annotated_generation"), sep=",")

    if not os.path.isfile(OUT_DIR):
        out_df = generation_df.copy()
        out_df.insert(loc=0, column="text", value=pd.NA) # type: ignore
        current_feedback = 0
    else:
        out_df = pd.read_csv(OUT_DIR, sep=",")
        current_feedback = out_df["text"].last_valid_index() + 1 # type: ignore

    print("start of generation")

    while current_feedback < len(generation_df):
        elective_name = str(random.choice(elective_names))
        response = make_generation_request(
            current_feedback,
            generation_df,
            elective_name
        )
        print(current_feedback, "\n\n", response)
        out_df.at[current_feedback, "text"] = response
        if current_feedback % 10 == 0:
            out_df.to_csv(OUT_DIR, sep=",", index=False)
        time.sleep(delay)
        current_feedback += 1
    out_df.to_csv(OUT_DIR, sep=",", index=False)
    print("\nГОТОВО!")
