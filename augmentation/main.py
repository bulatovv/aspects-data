import sys
import config
import time
import shutil
import pandas as pd
from pathlib import Path
from simple_term_menu import TerminalMenu
from augmentation import run_augmentation
from generation import run_generation

def get_menu_option(menu_options: list[str]):
    terminal_menu = TerminalMenu(menu_options)
    menu_entry_index = terminal_menu.show()
    return menu_options[menu_entry_index] # type: ignore

def get_bleu_metrics(original_texts \
                     , texts):
    return 1.0

def get_labse_metrics(original_texts \
                     , texts):
    return 1.0

def make_report_augmentation(df: pd.DataFrame):
    report_dir = Path(config.DATA_DIRECTORY) / ("report_augmentation_" + str(int(time.time())))
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.PROMPT_AUGMENTATION_CLIENT_PATH, report_dir)
    shutil.copy(config.PROMPT_AUGMENTATION_SYSTEM_PATH, report_dir)
    df.to_csv(report_dir / "output.csv")
    with open(report_dir / "metrics.txt", "w") as metrics_file:
        metrics_file.write("BLEU: " + str(get_bleu_metrics(df["original_text"], df["text"])) \
            + "\nLABSE: " + str(get_labse_metrics(df["original_text"], df["text"])))

# def make_report_generation(df: pd.DataFrame):
#     report_dir = Path(config.DATA_DIRECTORY) / ("report_generation_" + str(int(time.time())))
#     report_dir.mkdir(parents=True, exist_ok=True)
#     shutil.copy(config.PROMPT_GENERATION_CLIENT_PATH, report_dir)
#     shutil.copy(config.PROMPT_GENERATION_SYSTEM_PATH, report_dir)
#     df.to_csv(report_dir / "output.csv")

def main():
    try:
        answer = get_menu_option(['аугментация', 'генерация'])
        if answer == 'аугментация':
            answer = get_menu_option(['сгенерировать', 'создать отчёт'])
            if answer == 'сгенерировать':
                rows_count = int(input('Введите число строк или 0: '))
                remove_out = input('Начать сначала? (y/n) ')
                if (remove_out in ['д', 'Д', 'y', 'Y']):
                    config.AUGMENTATION_OUT_PATH.unlink(missing_ok=True)
                df = run_augmentation(config.OPENAI_REQUEST_DELAY, rows_count)
                make_report_augmentation(df)
            elif answer == 'создать отчёт':
                df = pd.read_csv(config.AUGMENTATION_OUT_PATH, sep=",")
                make_report_augmentation(df[df['text'].notna()]) # type: ignore
        elif answer == 'генерация':
            answer = get_menu_option(['сгенерировать'])
            if answer == 'сгенерировать':
                rows_count = int(input('Введите число строк или 0: '))
                remove_out = input('Начать сначала? (y/n) ')
                if (remove_out in ['д', 'Д', 'y', 'Y']):
                    config.GENERATION_OUT_PATH.unlink(missing_ok=True)
                df = run_generation(config.OPENAI_REQUEST_DELAY, rows_count)
                # make_report_generation(df)
            # elif answer == 'создать отчёт':
            #     df = pd.read_csv(config.GENERATION_OUT_PATH, sep=",")
            #     make_report_generation()
    except (KeyboardInterrupt, TypeError):
        print("\nПока!\n")
        sys.exit(0)

main()
