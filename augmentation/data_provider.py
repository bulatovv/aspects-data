import os
import requests
from pathlib import Path

import config

class DataNotFoundException(Exception):
    def __init__(self, entry: str):
        super().__init__(entry + " not found!")

class DataProvider():

    _DATA_DIRECTORY = Path(config.DATA_DIRECTORY)

    _DATA_ENTRIES = {
        "annotated": {
            "filename": config.DATA_ANNOTATED_FILENAME,
            "url": config.DATA_ANNOTATED_URL
        },
        "elective_names": {
            "filename": config.DATA_ELECTIVE_NAMES_FILENAME,
            "url": config.DATA_ELECTIVE_NAMES_URL
        },
    }

    @staticmethod
    def _filename(entry: str):
        return DataProvider._DATA_ENTRIES[entry]["filename"]

    @staticmethod
    def _url(entry: str):
        return DataProvider._DATA_ENTRIES[entry]["url"]

    @staticmethod
    def _path(entry: str):
        return DataProvider._DATA_DIRECTORY / DataProvider._DATA_ENTRIES[entry]["filename"]

    @staticmethod
    def _download_data(entry: str):
        path = DataProvider._path(entry)
        if os.path.isfile(path):
            return
        DataProvider._DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        response = requests.get(DataProvider._url(entry))
        open(path, "wb").write(response.content)

    @staticmethod
    def get_path(entry: str):
        if entry not in DataProvider._DATA_ENTRIES:
            raise DataNotFoundException(entry)
        DataProvider._download_data(entry)
        return DataProvider._path(entry)
