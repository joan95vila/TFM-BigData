# -*- coding: utf-8 -*-

import pickle

from utilities import debug
from utilities.constants import *
from gensim.models.doc2vec import Doc2Vec as Doc2vec_gensim, TaggedDocument
from nltk.tokenize import word_tokenize
import multiprocessing
from processing.text import load_dataset


class LearningSets:

    def __init__(self, embedding_model, train=False, verbose=True):
        training_sets = load_dataset.DatasetOperations(path=PATH_PROCESSED_DATASET)
        X_train, y_train, file_train, X_validation, y_validation, file_validation = \
            [set_ for set_ in training_sets.create_training_datasets()]
        self.files_train, self.files_validation = file_train, file_validation
        self.embedding = embedding_model

        if train:
            if verbose:
                print(f" > initializing ({embedding_model.embedding_params['embedding_name']}) EMBEDDING TRAINING "
                      f"with the configuration ({embedding_model.embedding_params['conf_name']}).")
                print(f"{' ' * 3}>> Training started.")

            self.embedding.train(X_train, save=True)

            if verbose:
                print(f"{' ' * 3}>> Training completed.")

            self.targets_train, self.feature_vectors_train = self.embedding.create_embedding_vectors(
                X_train, y_train=y_train, data_type="train")

            self.targets_validation, self.feature_vectors_validation = self.embedding.create_embedding_vectors(
                X_validation, y_train=y_validation, data_type="validation")

            if self.embedding.embedding_params["PCA"]:
                from sklearn.decomposition import PCA

                try:
                    if self.embedding.embedding_params['embedding_name'] == "Doc2Vec":
                        pca = PCA(n_components=self.embedding.embedding_params["vector_size"])
                    else:
                        pca = PCA(n_components=self.embedding.embedding_params["max_features"])
                    pc_feature_vectors_train = pca.fit_transform(self.feature_vectors_train)
                    pc_feature_vectors_validation = pca.transform(self.feature_vectors_validation)

                    self.feature_vectors_train = pc_feature_vectors_train
                    self.feature_vectors_validation = pc_feature_vectors_validation

                except ValueError:
                    print(f"{' ' * 3}>> The components of the PCA are larger than the features vector length passed, "
                          "hence no PCA are applied")

            path = PATH_SAVED_EMBEDDINGS + self.embedding.chunking + f"_embeddings_train.pickle"
            embedding_vectors = [self.targets_train, self.feature_vectors_train]
            pickle.dump(embedding_vectors, open(path, 'wb'))

            path = PATH_SAVED_EMBEDDINGS + self.embedding.chunking + f"_embeddings_validation.pickle"
            embedding_vectors = [self.targets_validation, self.feature_vectors_validation]
            pickle.dump(embedding_vectors, open(path, 'wb'))

        else:  # no train, then LOAD embeddings
            if verbose:
                print(f" > LOADING ({embedding_model.embedding_params['embedding_name']}) EMBEDDING "
                      f"with the configuration ({embedding_model.embedding_params['conf_name']}).")
                print(f"{' ' * 3}>> Started loading.")

            self.targets_train, self.feature_vectors_train = self.embedding.vectors_load(data_type="train")

            self.targets_validation, self.feature_vectors_validation = self.embedding.vectors_load(
                data_type="validation")

            if verbose:
                print(f"{' ' * 3}>> Completed loading.")

    def sentence_to_vector(self, sentence):
        self.embedding.sentence_to_vector(sentence)

    def visualize_3d(self):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(self.feature_vectors_train)
        x, y, z = principal_components[:, 0], principal_components[:, 1], principal_components[:, 2]

        title = "VISUAL DATA INFORMATION"
        body = f"Total variance explained in 3D: {np.sum(np.array(pca.explained_variance_ratio_))}\n" \
               f"Variance explained in 3D per dimension: {pca.explained_variance_ratio_}"
        debug.information_block(title, body)

        categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
        categories_map = {category: i for i, category in enumerate(categories)}
        invert_categories_map = {v: k for k, v in categories_map.items()}
        labels = [invert_categories_map[tag_index] for tag_index in self.targets_train]
        colours = ListedColormap(['violet', 'gold', 'dodgerblue', 'lawngreen', 'darkorange'])

        scatter = ax.scatter(x, y, z, c=self.targets_train, marker='o', cmap=colours)
        plt.legend(handles=scatter.legend_elements()[0], labels=labels)
        plt.show()


class Doc2Vec:

    def __init__(self, params):
        self.embedding_params = params
        self.chunking = f"{params['embedding_name']}_{params['conf_name']}"
        self.embedding = None

    def train(self, docs, save):
        # prepare data for training
        tagged_data = [TaggedDocument(word_tokenize(d), [i]) for i, d in enumerate(docs)]

        # create the embedding
        workers = multiprocessing.cpu_count()
        self.embedding = Doc2vec_gensim(workers=workers, vector_size=self.embedding_params["vector_size"],
                                        min_count=self.embedding_params["min_count"],
                                        window=self.embedding_params["window"],
                                        epochs=self.embedding_params["epochs"])
        self.embedding.build_vocab(tagged_data)  # ¿optional?
        self.embedding.train(tagged_data, total_examples=self.embedding.corpus_count, epochs=self.embedding.epochs)

        if save:
            self.__save_embedding_model()

    # PATH_SAVED_EMBEDDINGS
    def __save_embedding_model(self):
        file_name = PATH_SAVED_EMBEDDINGS + self.chunking + ".pickle"
        pickle.dump([self.embedding, self.embedding_params], open(file_name, 'wb'))

    def load_embedding(self):
        file_name = PATH_SAVED_EMBEDDINGS + self.chunking + ".pickle"
        self.embedding, self.embedding_params = pickle.load(open(file_name, 'rb'))

    def create_embedding_vectors(self, X_train, y_train, data_type=""):
        tagged_data = [TaggedDocument(word_tokenize(d), [y_train[i]]) for i, d in enumerate(X_train)]

        targets, feature_vectors = zip(*[(doc.tags[0], self.embedding.infer_vector(doc.words, epochs=20))
                                         for doc in tagged_data])

        embedding_vectors = targets, feature_vectors
        return embedding_vectors

    def vectors_load(self, data_type=""):
        self.load_embedding()

        path = PATH_SAVED_EMBEDDINGS + self.chunking + "_embeddings_" + data_type + ".pickle"
        return pickle.load(open(path, 'rb'))

    def sentence_to_vector(self, sentence):
        return self.embedding.infer_vector(word_tokenize(sentence), epochs=20)


class OneHotEncoding:

    def __init__(self, params):
        self.embedding_params = params
        self.chunking = f"{params['embedding_name']}_{params['conf_name']}"
        self.embedding = None

    def train(self, docs, save):
        print("WARNING: There is nothing to train on this embedding.")

    def vectors_load(self, data_type=""):
        path = PATH_SAVED_EMBEDDINGS + self.chunking + "_embeddings_" + data_type + ".pickle"
        return pickle.load(open(path, 'rb'))

    @staticmethod
    def sentence_to_vector(sentence):
        print("Method still not built.")
        return


class OneHotVocabularyEncoding(OneHotEncoding):

    def __init__(self, params):
        super().__init__(params)
        self.vocab = None

    def create_embedding_vectors(self, X_train, y_train, data_type=""):
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(vocabulary=self.vocab, max_features=3000)
        feature_vectors = vectorizer.fit_transform(X_train).toarray()

        feature_vectors = [list(map(lambda value: 1 if value > 0 else 0, doc_vec)) for doc_vec in feature_vectors]

        # print([len(feature_vectors), len(feature_vectors[0])])

        if data_type == "train":
            self.vocab = vectorizer.vocabulary_

        embedding_vectors = y_train, feature_vectors
        return embedding_vectors


class OneHotCountWordsEncoding(OneHotEncoding):
    def __init__(self, params):
        super().__init__(params)
        self.vocab = None

    def create_embedding_vectors(self, X_train, y_train, data_type=""):
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(vocabulary=self.vocab, max_features=3000)
        # vectorizer2 = CountVectorizer(ngram_range=(2, 2)) # (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams
        feature_vectors = vectorizer.fit_transform(X_train).toarray()

        # print(feature_vectors.shape)

        if data_type == "train":
            self.vocab = vectorizer.vocabulary_

        embedding_vectors = y_train, feature_vectors
        return embedding_vectors


class OneHotBoWEncoding(OneHotEncoding):
    def __init__(self, params):
        super().__init__(params)
        self.max_len = None

    def create_embedding_vectors(self, X_train, y_train, data_type=""):
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        import numpy as np

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        # Getting the biggest sentence
        if data_type == "train":
            self.max_len = np.max([len(doc.split()) for doc in X_train])

        feature_vectors = tokenizer.texts_to_sequences(X_train)
        feature_vectors = pad_sequences(feature_vectors, maxlen=self.max_len).tolist()

        embedding_vectors = y_train, feature_vectors
        return embedding_vectors


class TfidfVectorizer:
    def __init__(self, params):
        self.embedding_params = params
        self.chunking = f"{params['embedding_name']}_{params['conf_name']}"
        self.embedding = None

    def train(self, docs, save):
        from sklearn.feature_extraction.text import TfidfVectorizer as sklearn_TfidfVectorizer

        # create the embedding
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if self.embedding_params["PCA"]:
                self.embedding = sklearn_TfidfVectorizer(max_features=self.embedding_params["max_features"])
            else:
                self.embedding = sklearn_TfidfVectorizer(max_features=None)

        self.embedding.fit(docs)
        if save:
            self.__save_embedding_model()

    def __save_embedding_model(self):
        file_name = PATH_SAVED_EMBEDDINGS + self.chunking + ".pickle"
        pickle.dump([self.embedding, self.embedding_params], open(file_name, 'wb'))

    def load_embedding(self):
        file_name = PATH_SAVED_EMBEDDINGS + self.chunking + ".pickle"
        self.embedding, self.embedding_params = pickle.load(open(file_name, 'rb'))

    def create_embedding_vectors(self, X_train, y_train, data_type=""):
        feature_vectors = self.embedding.transform(X_train).toarray()

        embedding_vectors = y_train, feature_vectors
        return embedding_vectors

    def vectors_load(self, data_type=""):
        path = PATH_SAVED_EMBEDDINGS + self.chunking + "_embeddings_" + data_type + ".pickle"
        return pickle.load(open(path, 'rb'))

    @staticmethod
    def sentence_to_vector(sentence):
        print("Method still not built.")
        return
