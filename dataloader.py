"""Useful functions to load data from json files"""

import json
from typing import Dict, List, Tuple


def load_data(dataset_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Load both contexts and questions from a json file in SQuAD format.

    Parameters
    ----------
    dataset_path : str
        path to the json file containing the dataset

    Returns
    -------
    contexts, questions: Tuple[List[str], List[Dict]]
        the contexts and questions in the dataset.
        The questions are dictionaries such as {'question': '...', 'context_id': 0}
    """

    with open(dataset_path, "rb") as file:
        data = json.load(file)

    contexts = []
    questions = []
    i = 0
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            contexts.append(paragraph["context"])
            for qa in paragraph["qas"]:
                questions.append({"question": qa["question"], "context_id": i})

            i += 1

    return contexts, questions


def load_contexts(dataset_path: str) -> List[str]:
    """
    Load contexts from a json file in SQuAD format.

    Parameters
    ----------
    dataset_path : str
        path to the json file containing the dataset

    Returns
    -------
    contexts: List[str]
        the contexts in the dataset.
    """

    with open(dataset_path, "rb") as file:
        data = json.load(file)

    contexts = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            contexts.append(paragraph["context"])

    return contexts


def load_questions(dataset_path: str) -> List[Dict]:
    """
    Load questions from a json file in SQuAD format.

    Parameters
    ----------
    dataset_path : str
        path to the json file containing the dataset

    Returns
    -------
    questions: List[Dict]
        the questions in the dataset. The questions are dictionaries such as {'question': '...', 'context_id': 0}
    """

    with open(dataset_path, "rb") as file:
        data = json.load(file)

    questions = []
    i = 0
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                questions.append({"question": qa["question"], "context_id": i})
            i += 1

    return questions
