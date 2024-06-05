import os
import time
import pandas as pd
import openai_client
import config
from utilities import read_file
from data_provider import DataProvider

PROMPT_SYSTEM = read_file(config.PROMPT_AUGMENTATION_SYSTEM_PATH)
PROMPT_CLIENT = read_file(config.PROMPT_AUGMENTATION_CLIENT_PATH)
OUT_DIR = config.AUGMENTATION_OUT_PATH

def make_augmentation_request(current_feedback: int,
                 annotated_df: pd.DataFrame) -> str | None:
    label_to_sentiment = {
        1: "нейтральный",
        2: "позитивный",
        3: "негативный",
    }
    sentiments = [ i.replace("__", " и ").capitalize() + ": " + label_to_sentiment[v] for i, v in annotated_df.iloc[current_feedback].dropna().drop("text").items() ]
    response = openai_client.gpt_request(
        system_prompt=PROMPT_SYSTEM,
        client_prompt=PROMPT_CLIENT.format(
            original_feedback=annotated_df['text'][current_feedback],
            sentiment = "\n".join(sentiments)
        ),
        assistant_prompt="Вот перефразированный текст:\n",
    )
    return response

def run_augmentation(delay: float, rows_count: int) -> pd.DataFrame:
    annotated_df = pd.read_csv(DataProvider.get_path("annotated"), sep=",")

    if not os.path.isfile(OUT_DIR):
        out_df = annotated_df.copy()
        out_df.rename(columns={'text': 'original_text'}, inplace=True)
        out_df.insert(loc=0, column="text", value=pd.NA) # type: ignore
        current_feedback = 0
    else:
        out_df = pd.read_csv(OUT_DIR, sep=",")
        current_feedback = out_df["text"].last_valid_index() + 1 # type: ignore

    print("start of augmentation")

    while current_feedback < len(annotated_df) \
            and (rows_count == 0 \
                or current_feedback < rows_count):
        response = make_augmentation_request(current_feedback, annotated_df)
        print("\n\n", current_feedback, "\n", response)
        out_df.at[current_feedback, "text"] = response
        if current_feedback % 10 == 0:
            out_df.to_csv(OUT_DIR, sep=",", index=False)
        time.sleep(delay)
        current_feedback += 1
    out_df.to_csv(OUT_DIR, sep=",", index=False)
    print("\nГОТОВО!")

    if rows_count > 0:
        out_df = out_df.head(rows_count)

    return out_df
