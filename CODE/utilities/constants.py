#!/usr/bin/env python
# encoding: utf-8
"""
Constants.py
"""

from utilities.param_grid_conf import param_grid
from utilities.embeddings_params import embedding_params

CATEGORIES = {i: category for i, category in enumerate(['business', 'entertainment', 'politics', 'sport', 'tech'])}

# EMBEDDING_NAMES = ["Doc2Vec"]
# EMBEDDING_NAMES = ["OneHotVocabularyEncoding"]
# EMBEDDING_NAMES = ["OneHotCountWordsEncoding"]
# EMBEDDING_NAMES = ["OneHotBoWEncoding"]
# EMBEDDING_NAMES = ["TfidfVectorizer"]

# EMBEDDING_NAMES = ["OneHotVocabularyEncoding", "OneHotCountWordsEncoding"]

# EMBEDDING_NAMES = ["Doc2Vec", "TfidfVectorizer", "OneHotBoWEncoding"]
# EMBEDDING_NAMES = ["Doc2Vec", "TfidfVectorizer"]

EMBEDDING_NAMES = list(embedding_params.keys())
CLASSIFIER_NAMES = list(param_grid.keys())

PATH_PROCESSED_DATASET = "../DATASETS/BBC News Summary/News Articles Processed\\"
PATH_SAVED_EMBEDDINGS = "models\\SAVED EMBEDDINGS\\"
PATH_SAVED_CLASSIFIERS = "models\\SAVED CLASSIFIERS\\"
PATH_EXECUTION_TIMINGS = "models\\EXECUTION TIMINGS\\"
