"""Define BestContextFinder class, which can be used to find the best contexts for a given query"""


from random import sample
from typing import Dict, List, Tuple

import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from dataloader import load_contexts, load_data


class BestContextFinder:
    """Find the best contexts for specific queries"""

    def __init__(self, dataset_path="squad1/train-v1.1.json", with_questions=False) -> None:
        """
        Create a BestContextFinder, to find relevant paragraphs for a query.

        Parameters
        ----------
        dataset_path : str, optional
            path to the dataset (must be a json file in SQuAD format), by default "squad1/train-v1.1.json"
        """

        if with_questions:
            self.contexts, self.questions = load_data(dataset_path)
        else:
            self.contexts = load_contexts(dataset_path)
            self.questions = None

        self.vectorizer = TfidfVectorizer(min_df=0, stop_words=stopwords.words("english"))
        self.paragraphs_vectors = self.vectorizer.fit_transform(self.contexts)

    def get_best_contexts(self, query: str) -> Tuple[List[str], List[int], List[float]]:
        """
        Rank the contexts according to their relevance to the query.

        Parameters
        ----------
        query : str
            Query to use to rank the contexts.

        Returns
        -------
        Tuple[List[str], List[int], List[float]]
            A tuple with 3 lists:
                - the contexts in order of relevance to the query
                - the corresponding indexes of the contexts in the original list
                - the similarity_scores of the contexts with the query (in order of relevance)
        """

        # Search the best contexts using cosine similarity
        question_vector = self.vectorizer.transform([query])
        similarity_scores: np.ndarray = cosine_similarity(question_vector, self.paragraphs_vectors)
        top_indexes = list(similarity_scores.argsort()[0][::-1])
        return (
            [self.contexts[i] for i in top_indexes],
            top_indexes,
            [similarity_scores[0][i] for i in top_indexes],
        )

    def evaluate(self, num_samples: int = 200, repeat: int = 1) -> Dict[str, List[float]]:
        """
        Evaluate the model on a random sample of queries from the dataset. The model must have been loaded with questions to use this function.

        Parameters
        ----------
        num_samples : int, optional
            number of samples to use for the evaluation, by default 200
        repeat : int, optional
            number of times to repeat the evaluation (useful to evaluate the variance), by default 1

        Returns
        -------
        Dict[str, List[float]]
            dictionary with two lists (each item in a list corresponds to a run of evaluation):
            - "mean_ranks": the mean rank of the best context for each query
                            (how is the best context ranked in our model)
            - "accuracies": the accuracies of the best contexts for each query
                            (how many times did the model find the best context)
        """

        if self.questions is None:
            raise ValueError("The model must have questions to use this function")

        mean_ranks = []
        accuracies = []
        for _ in range(repeat):
            questions = sample(self.questions, num_samples)
            ranks = []
            accuracies.append(0)
            for question in tqdm(questions):
                _, top_indexes, _ = self.get_best_contexts(question["question"])
                ranks.append(top_indexes.index(question["context_id"]))
                if top_indexes.index(question["context_id"]) == 0:
                    accuracies[-1] += 1

            accuracies[-1] /= num_samples
            mean_ranks.append(np.mean(ranks))

        return {"mean_ranks": mean_ranks, "accuracies": accuracies}
