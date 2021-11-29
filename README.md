# Illuin Technology Project: Retrieving SQuAD

Find the best contexts among a large dataset for a given query.

## Quick Start

    python3 get_best_context.py --help

## Parameters choice

Two methods are available to retrieve the best context for a given query: `--method=tfidf` or `--method=transformers`.

The transformers is the default method, as it provides better results than Tf-Idf. However, the first time using transformers can take a while to load, as it has to download the model, and encode the entire dataset. For the SQuAD training dataset, the encoding can take more than 15 minutes, depending on your computer.

However, the embeddings are then cached, which allows you to use the transformers method for the next queries without any loading time.

The embeddings for the train and dev SQuAD datasets are already provided in cache.
