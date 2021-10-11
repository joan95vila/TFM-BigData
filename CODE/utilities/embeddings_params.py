embedding_params = {
    "Doc2Vec": {
        "window": [2],
        "PCA": [True, False],
        "vector_size": [10, 100, 500],  # vector size, same size if PCA is True (param shared) # For KNN and other algorithms 3000 is too much.
        "min_count": [2],
        "epochs": [100],

        # done automatically, starts from top param to bottom, read left to right. -> conf_name
        # "conf_name": ["conf_" + str(n) for n in range(1, (2 * 2) + 1)],
        # done automatically -> embedding_name
        # "embedding_name": "Doc2Vec"
    },
    "OneHotVocabularyEncoding": {
        "ngram_range": (1, 1),
        # "PCA": [False, True],
        "PCA": [True, False],
        "max_features": [10, 100, 500],  # vector size, same size if PCA is True (param shared) # None ->ConvergenceWarning: Solver terminated early (max_iter=50)
        # "max_features": [None, 10],  # vector size, same size if PCA is True (param shared)

        # done automatically, starts from top param to bottom, read left to right. -> conf_name
        # "conf_name": ["conf_" + str(n) for n in range(1, (2 * 2) + 1)],
        # done automatically -> embedding_name
        # "embedding_name": "OneHotVocabularyEncoding"
    },
    "OneHotCountWordsEncoding": {
        "ngram_range": (1, 1),
        # "PCA": [False, True],
        "PCA": [True, False],
        "max_features": [10, 100, 500],  # vector size, same size if PCA is True (param shared)
        # "max_features": [None, 10],  # vector size, same size if PCA is True (param shared)

        # done automatically, starts from top param to bottom, read left to right. -> conf_name
        # "conf_name": ["conf_" + str(n) for n in range(1, (2 * 2) + 1)],
        # done automatically -> embedding_name
        # "embedding_name": "OneHotCountWordsEncoding"
    },
    "OneHotBoWEncoding": {
        # "PCA": [False, True],
        "PCA": [True, False],
        # None = self.max_len (No limit) -> max_features
        "max_features": [10, 100, 500],  # vector size, same size if PCA is True (param shared)
        # "max_features": [None, 10],  # vector size, same size if PCA is True (param shared)

        # done automatically, starts from top param to bottom, read left to right. -> conf_name
        # "conf_name": ["conf_" + str(n) for n in range(1, (2 * 2) + 1)],
        # done automatically -> embedding_name
        # "embedding_name": "OneHotBoWEncoding"
    },
    "TfidfVectorizer": {
        "ngram_range": (1, 1),
        # "PCA": [False],
        "PCA": [True, False],
        # "max_features": [1000],
        "max_features": [10, 100, 500],  # vector size, same size if PCA is True (param shared)

        # done automatically, starts from top param to bottom, read left to right. -> conf_name
        # "conf_name": ["conf_" + str(n) for n in range(1, (2 * 2) + 1)],
        # done automatically -> embedding_name
        # "embedding_name": "TfidfVectorizer"
    }
}


def create_conf_combinations():
    def __infinite_sequence():
        n = 1
        while True:
            yield str(n)
            n += 1

    params_combinations = {embedding_name: [] for embedding_name in list(embedding_params.keys())}
    for embedding_name in list(embedding_params.keys()):
        n = __infinite_sequence()
        original_params = embedding_params[embedding_name]
        if embedding_name != "Doc2Vec":
            for PCA in original_params["PCA"]:
                for vector_size in original_params["max_features"]:
                    params_pipe = original_params.copy()

                    params_pipe["embedding_name"] = embedding_name
                    params_pipe["PCA"] = PCA
                    params_pipe["max_features"] = vector_size
                    params_pipe["conf_name"] = "conf_" + next(n)

                    # params_combinations.append(params_pipe)
                    params_combinations[embedding_name].append(params_pipe)
        else:
            for PCA in original_params["PCA"]:
                for vector_size in original_params["vector_size"]:
                    for window in original_params["window"]:
                        for min_count in original_params["min_count"]:
                            for epochs in original_params["epochs"]:
                                params_pipe = original_params.copy()

                                params_pipe["embedding_name"] = embedding_name
                                params_pipe["PCA"] = PCA
                                params_pipe["vector_size"] = vector_size
                                params_pipe["conf_name"] = "conf_" + next(n)
                                params_pipe["window"] = window
                                params_pipe["min_count"] = min_count
                                params_pipe["epochs"] = epochs

                                # params_combinations.append(params_pipe)
                                params_combinations[embedding_name].append(params_pipe)

    # print(params_combinations)

    return params_combinations


EMBEDDING_PARAMS = create_conf_combinations()

# EMBEDDING_PARAMS = {
# "Doc2Vec": {
#         "window": [2],
#         "PCA": [True, False],
#         "vector_size": [None],  # vector size, same size if PCA is True (param shared) # For KNN and other algorithms 3000 is too much.
#         "min_count": [2],
#         "epochs": [100],
#
#         # done automatically, starts from top param to bottom, read left to right. -> conf_name
#         "conf_name": ["conf_7"],
#         # done automatically -> embedding_name
#         # "embedding_name": "Doc2Vec"
#     },
#     "OneHotVocabularyEncoding": {
#         "ngram_range": (1, 1),
#         # "PCA": [False, True],
#         "PCA": [True, False],
#         "max_features": [10, 100, 500],  # vector size, same size if PCA is True (param shared) # None ->ConvergenceWarning: Solver terminated early (max_iter=50)
#         # "max_features": [None, 10],  # vector size, same size if PCA is True (param shared)
#
#         # done automatically, starts from top param to bottom, read left to right. -> conf_name
#         "conf_name": ["conf_7"],
#         # done automatically -> embedding_name
#         # "embedding_name": "OneHotVocabularyEncoding"
#     },
#     "OneHotCountWordsEncoding": {
#         "ngram_range": (1, 1),
#         # "PCA": [False, True],
#         "PCA": [True, False],
#         "max_features": [10, 100, 500],  # vector size, same size if PCA is True (param shared)
#         # "max_features": [None, 10],  # vector size, same size if PCA is True (param shared)
#
#         # done automatically, starts from top param to bottom, read left to right. -> conf_name
#         "conf_name": ["conf_7"],
#         # done automatically -> embedding_name
#         # "embedding_name": "OneHotCountWordsEncoding"
#     },
#     "OneHotBoWEncoding": {
#         # "PCA": [False, True],
#         "PCA": [True, False],
#         # None = self.max_len (No limit) -> max_features
#         "max_features": [10, 100, 500],  # vector size, same size if PCA is True (param shared)
#         # "max_features": [None, 10],  # vector size, same size if PCA is True (param shared)
#
#         # done automatically, starts from top param to bottom, read left to right. -> conf_name
#         "conf_name": ["conf_7"],
#         # done automatically -> embedding_name
#         # "embedding_name": "OneHotBoWEncoding"
#     },
#     "TfidfVectorizer": {
#         "ngram_range": (1, 1),
#         # "PCA": [False],
#         "PCA": [True, False],
#         "max_features": [None],  # vector size, same size if PCA is True (param shared)
#
#         # done automatically, starts from top param to bottom, read left to right. -> conf_name
#         "conf_name": ["conf_7", "conf_8"],
#         # done automatically -> embedding_name
#         # "embedding_name": "TfidfVectorizer"
#     }
# }